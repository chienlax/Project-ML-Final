import sys
import time
import os
import psutil
import logging

from absl import app
from absl import flags

from k_means_constrained import KMeansConstrained
from sklearn.discriminant_analysis import StandardScaler
import numpy as np
import pandas as pd

from ortools.sat.python import cp_model

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, Birch
from sklearn.preprocessing import StandardScaler
from k_means_constrained import KMeansConstrained

from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils import calculate_distance_matrix
import numpy as np

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

FLAGS = flags.FLAGS
flags.DEFINE_string("input_dir", "input", "Input data directory")
flags.DEFINE_string("output_dir", "output", "Schedule output directory")
flags.DEFINE_integer("timeslot", 27, "Số ca thi")
flags.DEFINE_integer("num_of_available_room", 40, "Số phòng khả dụng")
flags.DEFINE_integer("student_per_room", 100, "Số sinh viên/ phòng") # for normal room
flags.DEFINE_integer("min_student_per_room", 10, "Số sv tối thiểu/phòng")
flags.DEFINE_string("exam", "2024_2", "Đợt thi")
flags.DEFINE_string("location", "HN", "Địa điểm thi")
flags.DEFINE_integer("n_jobs", -1, "n_jobs")
flags.DEFINE_integer("max_division", 4, "max_division") # value n = n-1 divisions

def apply_clustering_algorithms(X, min_hp_sv, n_jobs=1, weights=(0.5, 0.3, 0.2)):
    best_y = None
    best_nc = 0
    best_score = -1
    best_algorithm = None

    # Define clustering algorithms to test
    clustering_algorithms = [
        ('KMeans', KMeans),
        ('KMeansConstrained', KMeansConstrained),
        ('AgglomerativeClustering', AgglomerativeClustering),
        ('SpectralClustering', SpectralClustering),
        ('Birch', Birch),
        ('GMM', GaussianMixture),
        ('MiniBatchKMeans', MiniBatchKMeans),
        ('PAM', kmedoids)
    ]

    # Standardize data
    X_scaled = StandardScaler().fit_transform(X)

    # Iterate over clustering algorithms
    for algo_name, algo in clustering_algorithms:
        labels = None
        weighted_score = -1  # Initialize weighted score for each algorithm

        if algo_name == 'KMeans':
            for nc in range(2, min((X.shape[0] // min_hp_sv) + 1, 11)):
                kmeans = algo(n_clusters=nc, random_state=0, algorithm='elkan')
                labels = kmeans.fit_predict(X_scaled)
                weighted_score = calculate_weighted_score(X_scaled, labels, weights)
                if weighted_score > best_score:
                    best_y, best_nc, best_score, best_algorithm = labels, nc, weighted_score, algo_name

        elif algo_name == 'KMeansConstrained':
            for nc in range(2, min((X.shape[0] // min_hp_sv) + 1, 11)):
                kmeans = algo(n_clusters=nc, size_min=min_hp_sv, n_jobs=n_jobs)
                labels = kmeans.fit_predict(X_scaled)
                weighted_score = calculate_weighted_score(X_scaled, labels, weights)
                if weighted_score > best_score:
                    best_y, best_nc, best_score, best_algorithm = labels, nc, weighted_score, algo_name

        elif algo_name == 'AgglomerativeClustering':
            for nc in range(2, min((X.shape[0] // min_hp_sv) + 1, 11)):
                agglom = algo(n_clusters=nc)
                labels = agglom.fit_predict(X_scaled)
                weighted_score = calculate_weighted_score(X_scaled, labels, weights)
                if weighted_score > best_score:
                    best_y, best_nc, best_score, best_algorithm = labels, nc, weighted_score, algo_name

        elif algo_name == 'SpectralClustering':
            for nc in range(2, min((X.shape[0] // min_hp_sv) + 1, 11)):
                spectral = algo(n_clusters=nc, random_state=0)
                labels = spectral.fit_predict(X_scaled)
                weighted_score = calculate_weighted_score(X_scaled, labels, weights)
                if weighted_score > best_score:
                    best_y, best_nc, best_score, best_algorithm = labels, nc, weighted_score, algo_name

        elif algo_name == 'Birch':
            for nc in range(2, min((X.shape[0] // min_hp_sv) + 1, 11)):
                birch = algo(n_clusters=nc)
                labels = birch.fit_predict(X_scaled)
                weighted_score = calculate_weighted_score(X_scaled, labels, weights)
                if weighted_score > best_score:
                    best_y, best_nc, best_score, best_algorithm = labels, nc, weighted_score, algo_name

        elif algo_name == 'GMM':
            for nc in range(2, min((X.shape[0] // min_hp_sv) + 1, 11)):
                gmm = algo(n_components=nc, random_state=0)
                labels = gmm.fit_predict(X_scaled)
                weighted_score = calculate_weighted_score(X_scaled, labels, weights)
                if weighted_score > best_score:
                    best_y, best_nc, best_score, best_algorithm = labels, nc, weighted_score, algo_name

        elif algo_name == 'MiniBatchKMeans':
            for nc in range(2, min((X.shape[0] // min_hp_sv) + 1, 11)):
                minibatch_kmeans = algo(n_clusters=nc, random_state=0, batch_size=256)
                labels = minibatch_kmeans.fit_predict(X_scaled)
                weighted_score = calculate_weighted_score(X_scaled, labels, weights)
                if weighted_score > best_score:
                    best_y, best_nc, best_score, best_algorithm = labels, nc, weighted_score, algo_name

        elif algo_name == 'PAM':
            distance_matrix = calculate_distance_matrix(X_scaled)
            for nc in range(2, min((X.shape[0] // min_hp_sv) + 1, 11)):
                initial_medoids = list(range(nc))  # Select initial medoids
                pam = algo(distance_matrix, initial_medoids)
                pam.process()

                clusters = pam.get_clusters()
                labels = np.zeros(X_scaled.shape[0], dtype=int)
                for cluster_id, cluster_indices in enumerate(clusters):
                    for index in cluster_indices:
                        labels[index] = cluster_id

                weighted_score = calculate_weighted_score(X_scaled, labels, weights)
                if weighted_score > best_score:
                    best_y, best_nc, best_score, best_algorithm = labels, nc, weighted_score, algo_name

    return best_y, best_nc, best_score, best_algorithm

def calculate_weighted_score(X_scaled, labels, weights):
    silhouette = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else -1
    calinski_harabasz = calinski_harabasz_score(X_scaled, labels) if len(set(labels)) > 1 else -1
    davies_bouldin = davies_bouldin_score(X_scaled, labels) if len(set(labels)) > 1 else -1

    weighted_score = (weights[0] * silhouette +
                      weights[1] * calinski_harabasz -
                      weights[2] * davies_bouldin)
    
    return weighted_score

def export_report(X, min_hp_sv, output, dot_thi, tinh, weights=(0.5, 0.3, 0.2), n_jobs=1):
    # Set the report path
    clustering_report_path = f"{output}/{dot_thi}/{dot_thi}_{tinh}_clustering_report.csv"

    report_data = []
    
    # Define clustering algorithms to test
    clustering_algorithms = [
        ('KMeans', KMeans),
        ('KMeansConstrained', KMeansConstrained),
        ('AgglomerativeClustering', AgglomerativeClustering),
        ('SpectralClustering', SpectralClustering),
        ('Birch', Birch),
        ('GMM', GaussianMixture),
        ('MiniBatchKMeans', MiniBatchKMeans),
        ('PAM', kmedoids)
    ]

    # Standardize data
    X_scaled = StandardScaler().fit_transform(X)

    # Iterate over clustering algorithms
    for algo_name, algo in clustering_algorithms:
        # Track the best result for each algorithm
        labels = None
        
        if algo_name == 'KMeans':
            for nc in range(2, min((X.shape[0] // min_hp_sv) + 1, 11)):
                kmeans = algo(n_clusters=nc, random_state=0, n_jobs=n_jobs)
                labels = kmeans.fit_predict(X_scaled)
                weighted_score = calculate_weighted_score(X_scaled, labels, weights)
                
                report_data.append({
                    'Algorithm': algo_name,
                    'n_clusters': nc,
                    'Silhouette Score': silhouette_score(X_scaled, labels),
                    'Calinski-Harabasz Score': calinski_harabasz_score(X_scaled, labels),
                    'Davies-Bouldin Score': davies_bouldin_score(X_scaled, labels),
                    'Weighted Score': weighted_score
                })

        elif algo_name == 'KMeansConstrained':
            for nc in range(2, min((X.shape[0] // min_hp_sv) + 1, 11)):
                kmeans = algo(n_clusters=nc, size_min=min_hp_sv, n_jobs=n_jobs)
                labels = kmeans.fit_predict(X_scaled)
                weighted_score = calculate_weighted_score(X_scaled, labels, weights)
                
                report_data.append({
                    'Algorithm': algo_name,
                    'n_clusters': nc,
                    'Silhouette Score': silhouette_score(X_scaled, labels),
                    'Calinski-Harabasz Score': calinski_harabasz_score(X_scaled, labels),
                    'Davies-Bouldin Score': davies_bouldin_score(X_scaled, labels),
                    'Weighted Score': weighted_score
                })

        elif algo_name == 'AgglomerativeClustering':
            for nc in range(2, min((X.shape[0] // min_hp_sv) + 1, 11)):
                agglom = algo(n_clusters=nc)
                labels = agglom.fit_predict(X_scaled)
                weighted_score = calculate_weighted_score(X_scaled, labels, weights)
                
                report_data.append({
                    'Algorithm': algo_name,
                    'n_clusters': nc,
                    'Silhouette Score': silhouette_score(X_scaled, labels),
                    'Calinski-Harabasz Score': calinski_harabasz_score(X_scaled, labels),
                    'Davies-Bouldin Score': davies_bouldin_score(X_scaled, labels),
                    'Weighted Score': weighted_score
                })
                
        elif algo_name == 'SpectralClustering':
            for nc in range(2, min((X.shape[0] // min_hp_sv) + 1, 11)):
                spectral = algo(n_clusters=nc, random_state=0)
                labels = spectral.fit_predict(X_scaled)
                weighted_score = calculate_weighted_score(X_scaled, labels, weights)
                
                report_data.append({
                    'Algorithm': algo_name,
                    'n_clusters': nc,
                    'Silhouette Score': silhouette_score(X_scaled, labels),
                    'Calinski-Harabasz Score': calinski_harabasz_score(X_scaled, labels),
                    'Davies-Bouldin Score': davies_bouldin_score(X_scaled, labels),
                    'Weighted Score': weighted_score
                })
                

        elif algo_name == 'Birch':
            for nc in range(2, min((X.shape[0] // min_hp_sv) + 1, 11)):
                birch = algo(n_clusters=nc)
                labels = birch.fit_predict(X_scaled)
                weighted_score = calculate_weighted_score(X_scaled, labels, weights)
                report_data.append({
                    'Algorithm': algo_name,
                    'n_clusters': nc,
                    'Silhouette Score': silhouette_score(X_scaled, labels),
                    'Calinski-Harabasz Score': calinski_harabasz_score(X_scaled, labels),
                    'Davies-Bouldin Score': davies_bouldin_score(X_scaled, labels),
                    'Weighted Score': weighted_score
                })

        elif algo_name == 'GMM':
            for nc in range(2, min((X.shape[0] // min_hp_sv) + 1, 11)):
                gmm = algo(n_components=nc, random_state=0)
                labels = gmm.fit_predict(X_scaled)
                weighted_score = calculate_weighted_score(X_scaled, labels, weights)
                
                report_data.append({
                    'Algorithm': algo_name,
                    'n_clusters': nc,
                    'Silhouette Score': silhouette_score(X_scaled, labels),
                    'Calinski-Harabasz Score': calinski_harabasz_score(X_scaled, labels),
                    'Davies-Bouldin Score': davies_bouldin_score(X_scaled, labels),
                    'Weighted Score': weighted_score
                })

        elif algo_name == 'MiniBatchKMeans':
            for nc in range(2, min((X.shape[0] // min_hp_sv) + 1, 11)):
                minibatch_kmeans = algo(n_clusters=nc, random_state=0, batch_size=256)
                labels = minibatch_kmeans.fit_predict(X_scaled)
                weighted_score = calculate_weighted_score(X_scaled, labels, weights)
                
                report_data.append({
                    'Algorithm': algo_name,
                    'n_clusters': nc,
                    'Silhouette Score': silhouette_score(X_scaled, labels),
                    'Calinski-Harabasz Score': calinski_harabasz_score(X_scaled, labels),
                    'Davies-Bouldin Score': davies_bouldin_score(X_scaled, labels),
                    'Weighted Score': weighted_score
                })

        elif algo_name == 'PAM':
            distance_matrix = calculate_distance_matrix(X_scaled)
            for nc in range(2, min((X.shape[0] // min_hp_sv) + 1, 11)):
                initial_medoids = list(range(nc))  # Select initial medoids
                pam = algo(distance_matrix, initial_medoids)
                pam.process()

                clusters = pam.get_clusters()
                labels = np.zeros(X_scaled.shape[0], dtype=int)
                for cluster_id, cluster_indices in enumerate(clusters):
                    for index in cluster_indices:
                        labels[index] = cluster_id

                weighted_score = calculate_weighted_score(X_scaled, labels, weights)
                
                report_data.append({
                    'Algorithm': algo_name,
                    'n_clusters': nc,
                    'Silhouette Score': silhouette_score(X_scaled, labels),
                    'Calinski-Harabasz Score': calinski_harabasz_score(X_scaled, labels),
                    'Davies-Bouldin Score': davies_bouldin_score(X_scaled, labels),
                    'Weighted Score': weighted_score
                })

    report_df = pd.DataFrame(report_data)
    
    report_df.to_csv(clustering_report_path, index=False)
    logging.info(f"Clustering report saved to: {clustering_report_path}")
    logging.info(f"Clustering report data:\n{report_df}")

def main(argv):
    FLAGS(argv)
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

    folder_out = os.path.join(output_dir, dot_thi)
    os.makedirs(folder_out, exist_ok=True)

    # Configure logging
    log_file = os.path.join(folder_out, 'log.txt')
    
    # Remove any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    # Configure logging with both file and console output
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Starting execution with exam: {dot_thi}, location: {tinh}")

    file_path = f"{input_dir}/{dot_thi}/{dot_thi}_{tinh}.csv"
    data = pd.read_csv(file_path, encoding='utf-8-sig')
    data["Time slot"] = -1
    logging.debug("Initialized Time slot column with -1")

    while True:
        cm = pd.crosstab(data["Ma SV"], data["Ma HP"])
        conflict = cm.T.dot(cm)
        logging.debug(f"\n{conflict}")
        logging.debug(f"\n{data}")
        
        sv = data["Ma SV"].unique()
        hp = data.groupby("Ma HP").agg({"Ma SV": "count", "Time slot": "first"})
        hp = hp.rename(columns={"Ma SV": "sv"})
        hp["rooms"] = np.ceil(hp["sv"] / student_per_room).astype(int)
        logging.debug(f"\n{sv}")

        logging.debug(f"Unique students:\n{sv}")
        logging.debug(f"Grouped data:\n{hp}")
        
        model = cp_model.CpModel()
        x = {}
        logging.info("Model and variables initialized")

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
        solver.parameters.max_time_in_seconds = 120
        solver.parameters.log_search_progress = True
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            logging.info(f"Objective value: {solver.ObjectiveValue()}")

            for t in range(n_time_slot):
                logging.info(f"Time slot {t}: ")
                logging.info(
                    f"Number of rooms: {sum((solver.Value(x[(i, t)]) * hp.loc[i, 'rooms']) for i in hp.index)}"
                )
                for i in hp.index:
                    if solver.Value(x[(i, t)]) > 0:
                        data.loc[data["Ma HP"] == i, "Time slot"] = t
                        logging.info(f"Exam {i} scheduled in time slot {t}")

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
                        logging.info(
                            f"Conflict between {i} ({hp.loc[i, 'sv']}) and {j} ({hp.loc[j, 'sv']}): {conflict.loc[i, j]}"
                        )

            for i in split_set:
                if hp.loc[i, "sv"] >= min_hp_sv * 2:
                    X = cm.loc[cm[i] == 1].drop(i, axis=1)
                    X = X.loc[:, (X.max(axis=0) > 0)]
                    X = X.loc[:, (X.min(axis=0) < 1)]
                    if X.shape[1] == 0:
                        continue

                    best_y, best_nc, best_score, best_algorithm = apply_clustering_algorithms(X, min_hp_sv, n_jobs=n_jobs)
        
                    # Log the best clustering result
                    if best_y is not None:
                        logging.info(f"Splitting {i} ({hp.loc[i, 'sv']}) into {best_nc} clusters using {best_algorithm} with score {best_score}")
                        
                    X["cluster"] = best_y

                    for s in range(best_nc):
                        s_i = X.loc[X["cluster"] == s].index.tolist()
                        data.loc[(data["Ma HP"] == i) & (data["Ma SV"].isin(s_i)), "Ma HP"] = f"{i}_{s}"

            if solver.value(obj) == 0:
                logging.info("Solution found")
                
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
                    logging.info(f"Time slot {t}: ")
                    logging.info(
                        f"Number of rooms: {sum((solver.value(x[(i, t)]) * hp.loc[i, 'rooms']) for i in hp.index)}"
                    )

                    for i in hp.index:
                        if solver.value(x[(i, t)]) > 0:
                            data.loc[data["Ma HP"] == i, "Time slot"] = t
                            logging.info(f"Exam {i}")
                break
        else:
            logging.error("No solution found")
            return

    for i in hp.index:
        data.loc[data["Ma HP"] == i, "Ma HP"] = i.split("_")[0]

    data = data.sort_values(
        by=["Time slot", "Ma SV", "Ma HP"],
        ascending=[True, True, True],
    )

    folder_out = "/".join([output_dir, dot_thi])
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)

    data.to_csv(f"{folder_out}/{dot_thi}_{tinh}.csv", index=False, encoding='utf-8-sig', mode='w+')
    data.to_csv(f"{folder_out}/{dot_thi}_{tinh}_raw.csv", index=False, encoding='utf-8-sig', mode='w+')

if __name__ == "__main__":
    p = psutil.Process()
    p.cpu_affinity([1, 2, 3, 4, 5])
    app.run(main)
    