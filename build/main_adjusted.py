import sys
import time
import os
import psutil
import logging

from absl import app
from absl import flags

from k_means_constrained import KMeansConstrained
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd

from ortools.sat.python import cp_model

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering, AffinityPropagation, MeanShift, OPTICS, Birch#, GaussianMixture
from sklearn import mixture

FLAGS = flags.FLAGS
flags.DEFINE_string("input_dir", "input", "Input data directory")
flags.DEFINE_string("output_dir", "output", "Schedule output directory")
flags.DEFINE_integer("timeslot", 24, "Số ca thi")
flags.DEFINE_integer("num_of_available_room", 40, "Số phòng khả dụng")
flags.DEFINE_integer("student_per_room", 40, "Số sinh viên/ phòng") # for normal room
flags.DEFINE_integer("min_student_per_room", 15, "Số sv tối thiểu/phòng")
flags.DEFINE_string("exam", "2024_1", "Đợt thi")
flags.DEFINE_string("location", "HN", "Địa điểm thi")
flags.DEFINE_integer("n_jobs", -1, "n_jobs")
flags.DEFINE_integer("max_division", 5, "max_division") # value n = n-1 divisions

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def apply_clustering(X, min_sv_hp, max_clusters=10, n_jobs=1, output_report_path=None):
    X_scaled = StandardScaler().fit_transform(X)
    
    clustering_results = {}
    best_algorithm = None
    best_labels = None
    best_score = -1

    # Define clustering models, including KMeansConstrained
    clustering_algorithms = {
        "KMeans": KMeans(n_clusters=min(max_clusters, X.shape[0] // min_sv_hp), n_init=10, random_state=42),
        "KMeansConstrained": KMeansConstrained(
            n_clusters=min(max_clusters, X.shape[0] // min_sv_hp),
            size_min=min_sv_hp,
            n_jobs=n_jobs,
            random_state=0
        ),
        "AffinityPropagation": AffinityPropagation(damping=0.9),
        "MeanShift": MeanShift(bin_seeding=True),
        "OPTICS": OPTICS(min_samples=min_sv_hp, n_jobs=n_jobs),
        "Birch": Birch(n_clusters=min(max_clusters, X.shape[0] // min_sv_hp)),
        "AgglomerativeClustering": AgglomerativeClustering(
            n_clusters=min(max_clusters, X.shape[0] // min_sv_hp)
        ),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=min_sv_hp),
        "SpectralClustering": SpectralClustering(
            n_clusters=min(max_clusters, X.shape[0] // min_sv_hp), affinity="nearest_neighbors", n_jobs=n_jobs
        )
    }
    
    # Loop over each algorithm
    for name, algorithm in clustering_algorithms.items():
        try:
            # For GaussianMixture, fit_predict is not available, so we use predict after fit
            if name == "GaussianMixture":
                labels = algorithm.fit(X_scaled).predict(X_scaled)
            else:
                labels = algorithm.fit_predict(X_scaled)
            
            # Only evaluate silhouette score if we have more than one cluster
            if len(set(labels)) > 1:  
                score = silhouette_score(X_scaled, labels)
                clustering_results[name] = {"score": score, "labels": labels}

                # Update best clustering based on silhouette score
                if score > best_score:
                    best_algorithm = name
                    best_labels = labels
                    best_score = score
        except Exception as e:
            clustering_results[name] = {"score": None, "labels": None, "error": str(e)}
            continue

    # Generate a report and save it
    if output_report_path:
        with open(output_report_path, 'w') as report_file:
            report_file.write("Clustering Algorithm Performance Report\n")
            report_file.write("=" * 40 + "\n")
            for algo_name, result in clustering_results.items():
                report_file.write(f"Algorithm: {algo_name}\n")
                if result["score"] is not None:
                    report_file.write(f"  - Silhouette Score: {result['score']}\n")
                    report_file.write(f"  - Number of Clusters: {len(set(result['labels']))}\n")
                else:
                    report_file.write(f"  - Error: {result['error']}\n")
                report_file.write("\n")

            report_file.write(f"Best Algorithm: {best_algorithm}\n")
            report_file.write(f"Best Silhouette Score: {best_score}\n")
            report_file.write("=" * 40 + "\n")

    return best_algorithm, best_labels, best_score

def main(argv):
    dot_thi = FLAGS.exam
    tinh = FLAGS.location
    time_slot = FLAGS.timeslot
    n_avai_room = FLAGS.num_of_available_room
    student_per_room = FLAGS.student_per_room
    min_sv_hp = FLAGS.min_student_per_room
    n_jobs = FLAGS.n_jobs
    input_dir = FLAGS.input_dir
    output_dir = FLAGS.output_dir
    max_division = FLAGS.max_division

    file_path = f"{input_dir}/{dot_thi}/{dot_thi}_{tinh}.csv"
    clustering_report_path = f"{input_dir}/{dot_thi}/{dot_thi}_{tinh}_clustering_report.csv"
    data = pd.read_csv(file_path, encoding='utf-8-sig')
    data = data[data["Bộ môn giảng dạy"] != "Bộ môn Giáo dục thể chất"]
    data["Time slot"] = -1

    while True:
        cm = pd.crosstab(data["Mã SV"], data["Mã HP"])
        conflict = cm.T.dot(cm)
        logger.debug(f"\n{conflict}")
        logger.debug(f"\n{data}")
        
        sv = data["Mã SV"].unique()
        hp = data.groupby("Mã HP").agg({"Mã SV": "count", "Time slot": "first"})
        hp = hp.rename(columns={"Mã SV": "sv"})
        hp["rooms"] = np.ceil(hp["sv"] / student_per_room).astype(int)
        logger.debug(f"\n{sv}")
        
        model = cp_model.CpModel()
        x = {}
        n_time_slot = time_slot
        min_hp_sv = min_sv_hp
        half_day = time_slot // 2
        n_room = ([n_avai_room] * half_day + [n_avai_room] * half_day) * 2

        for i in hp.index:
            for j in range(n_time_slot):
                x[(i, j)] = model.new_bool_var(name=f"x_{i}_{j}")
                if hp.loc[i, "Time slot"] != -1:
                    if hp.loc[i, "Time slot"] == j:
                        model.add_hint(x[(i, j)], 1)
                    else:
                        model.add_hint(x[(i, j)], 0)

        for i in hp.index:
            model.add(sum(x[(i, j)] for j in range(n_time_slot)) == 1)

        for j in range(n_time_slot):
            model.add(
                sum((x[(i, j)] * hp.loc[i, "sv"]) for i in hp.index) <= n_room[j] * student_per_room
            )

        obj = 0
        same = {}
        s = {}

        for i in hp.index:
            for j in hp.index:
                if i < j and conflict.loc[i, j] > 0:
                    if (
                        hp.loc[i, "sv"] < min_hp_sv * 2
                        and hp.loc[j, "sv"] < min_hp_sv * 2
                    ) or (
                        len(i.split("_")) >= max_division
                        or len(j.split("_")) >= max_division
                    ):
                        for k in range(n_time_slot):
                            model.add(x[(i, k)] + x[(j, k)] <= 1)
                    else:
                        same[(i, j)] = model.new_bool_var(name=f"same_{i}_{j}")
                        if (
                            hp.loc[i, "Time slot"] != -1
                            and hp.loc[j, "Time slot"] != -1
                        ):
                            if hp.loc[i, "Time slot"] == hp.loc[j, "Time slot"]:
                                model.add_hint(same[(i, j)], 1)
                            else:
                                model.add_hint(same[(i, j)], 0)
                        for k in range(n_time_slot):
                            model.add(x[(i, k)] + x[(j, k)] <= 1 + same[(i, j)])
                    
                        obj += same[(i, j)] * (1 + conflict.loc[i, j])

        for i in hp.index:
            for j in hp.index:
                if i < j and i.split("_")[0] == j.split("_")[0]:
                    for k in range(0, n_time_slot, 3):
                        ai = sum(x[(i, k + t)] for t in range(3))
                        aj = sum(x[(j, k + t)] for t in range(3))
                        model.add(ai == aj)

        model.minimize(obj)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60
        solver.parameters.log_search_progress = True
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            logger.info(f"Objective value: {solver.ObjectiveValue()}")

            for t in range(n_time_slot):
                logger.info(f"Time slot {t}: ")
                logger.info(
                    f"Number of rooms: {sum((solver.Value(x[(i, t)]) * hp.loc[i, 'rooms']) for i in hp.index)}"
                )
                for i in hp.index:
                    if solver.Value(x[(i, t)]) > 0:
                        data.loc[data["Mã HP"] == i, "Time slot"] = t
                        logger.info(f"Exam {i} scheduled in time slot {t}")

            split_set = set()
            for i in hp.index:
                for j in hp.index:
                    if (
                        i < j
                        and (
                            hp.loc[i, "sv"] >= min_hp_sv * 2
                            or hp.loc[j, "sv"] >= min_hp_sv * 2
                        )
                        and conflict.loc[i, j] > 0
                        and (len(i.split("_")) < max_division and len(j.split("_")) < max_division)
                        and solver.Value(same[(i, j)]) > 0
                    ):
                        split = i if (hp.loc[i, "sv"] < hp.loc[j, "sv"] or len(i.split("_")) > len(j.split("_"))) else j
                        split_set.add(split)
                        logger.info(
                            f"Conflict between {i} ({hp.loc[i, 'sv']}) and {j} ({hp.loc[j, 'sv']}): {conflict.loc[i, j]}"
                        )

            for i in split_set:
                if hp.loc[i, "sv"] >= min_hp_sv * 2:
                    X = cm.loc[cm[i] == 1].drop(i, axis=1)
                    X = X.loc[:, (X.max(axis=0) > 0)]
                    X = X.loc[:, (X.min(axis=0) < 1)]
                    if X.shape[1] == 0:
                        continue

                    # Use apply_clustering with a report path
                    best_algorithm, best_labels, best_score = apply_clustering(
                        X, min_sv_hp, max_clusters=11, n_jobs=n_jobs, output_report_path=clustering_report_path
                    )

                    if best_labels is not None:
                        X["cluster"] = best_labels
                        logger.info(f"Best clustering for {i}: {best_algorithm} with score {best_score}")

                        for s in np.unique(best_labels):
                            s_i = X[X["cluster"] == s].index.tolist()
                            data.loc[(data["Mã HP"] == i) & (data["Mã SV"].isin(s_i)), "Mã HP"] = f"{i}_{s}"

            if solver.value(obj) == 0:
                logger.info("Solution found")
                
                for i in hp.index:
                    for j in hp.index:
                        if i < j and conflict.loc[i, j] > 0:
                            for k in range(n_time_slot):
                                model.add(x[(i, k)] + x[(j, k)] <= 1)

                for v in same.values():
                    model.add(v == 0)

                model.clear_hints()
                model.clear_objective()

                for i in hp.index:
                    for j in range(n_time_slot):
                        if hp.loc[i, "Time slot"] != -1:
                            if hp.loc[i, "Time slot"] == j:
                                model.add_hint(x[(i, j)], 1)
                            else:
                                model.add_hint(x[(i, j)], 0)

                y = {}
                for i in hp.index:
                    if len(i.split("_")) > 1:
                        for j in range(n_time_slot):
                            k = (i.split("_")[0], j)
                            if k not in y:
                                y[k] = model.new_bool_var(name=f"y_{i}_{j}")
                        model.add(
                            sum(y[(i.split("_")[0], j)] for j in range(n_time_slot))
                            <= 6
                        )

                for i in hp.index:
                    if len(i.split("_")) > 1:
                        for j in range(n_time_slot):
                            model.add(y[(i.split("_")[0], j)] >= x[(i, j)])

                start = {}
                end = {}
                for i in hp.index:
                    if len(i.split("_")) > 1:
                        if i.split("_")[0] not in start:
                            start[i.split("_")[0]] = model.new_int_var(
                                0, n_time_slot - 1, name=f"start_{i.split('_')[0]}"
                            )
                            end[i.split("_")[0]] = model.new_int_var(
                                0, n_time_slot - 1, name=f"end_{i.split('_')[0]}"
                            )
                            model.add(end[i.split("_")[0]] >= start[i.split("_")[0]])
                            model.add(
                                end[i.split("_")[0]] - start[i.split("_")[0]] <= 6
                            )

                for i in hp.index:
                    if len(i.split("_")) > 1:
                        for j in range(n_time_slot):
                            model.add(start[i.split("_")[0]] <= j).only_enforce_if(
                                x[(i, j)]
                            )
                            model.add(j <= end[i.split("_")[0]]).only_enforce_if(
                                x[(i, j)]
                            )

                obj2 = sum(y.values())
                obj2 += sum(end.values()) - sum(start.values())

                model.minimize(obj2)

                solver = cp_model.CpSolver()
                solver.parameters.max_time_in_seconds = 60  
                solver.parameters.log_search_progress = True
                status = solver.Solve(model)

                for t in range(n_time_slot):
                    logger.info(f"Time slot {t}: ")
                    logger.info(
                        f"Number of rooms: {sum((solver.value(x[(i, t)]) * hp.loc[i, 'rooms']) for i in hp.index)}"
                    )

                    for i in hp.index:
                        if solver.value(x[(i, t)]) > 0:
                            data.loc[data["Mã HP"] == i, "Time slot"] = t
                            logger.info(f"Exam {i}")
                break
        else:
            logger.error("No solution found")
            return

    for i in hp.index:
        data.loc[data["Mã HP"] == i, "Mã HP"] = i.split("_")[0]

    folder_out = "/".join([output_dir, dot_thi])
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
        
    data = data.sort_values(
        by=["Time slot", "Mã HP", "Mã LHP", "Khoa giảng dạy", "Tên", "Họ Lót", "Mã SV"],
        ascending=[True, True, True, True, True, True, True],
    )

    data.to_csv(f"{folder_out}/{dot_thi}_{tinh}.csv", index=False, encoding='utf-8-sig')
    data.to_csv(f"{folder_out}/{dot_thi}_{tinh}_raw.csv", index=False, encoding='utf-8-sig')
    
if __name__ == "__main__":
    p = psutil.Process()
    p.cpu_affinity([1, 2, 3, 4])
    app.run(main)
    