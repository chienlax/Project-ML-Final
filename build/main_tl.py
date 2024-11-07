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

FLAGS = flags.FLAGS
flags.DEFINE_string("input_dir", "input", "Input data directory")
flags.DEFINE_string("output_dir", "output", "Schedule output directory")
flags.DEFINE_integer("timeslot", 24, "Số ca thi")
flags.DEFINE_integer("num_of_available_room", 40, "Số phòng khả dụng")
flags.DEFINE_integer("student_per_room", 40, "Số sinh viên/ phòng") # for normal room
flags.DEFINE_integer("min_student_per_room", 10, "Số sv tối thiểu/phòng")
flags.DEFINE_string("exam", "2024_1", "Đợt thi")
flags.DEFINE_string("location", "HN", "Địa điểm thi")
flags.DEFINE_integer("n_jobs", -1, "n_jobs")
flags.DEFINE_integer("max_division", 4, "max_division") # value n = n-1 divisions

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

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

                    best_algorithm, best_labels, best_score = apply_clustering(X, min_hp_sv, max_clusters=11, n_jobs=n_jobs)

                    if best_labels is not None:
                        X["cluster"] = best_labels
                        logger.info(f"Best clustering for {i}: {best_algorithm} with score {best_score}")

                        for s in np.unique(best_labels):
                            s_i = X[X["cluster"] == s].index.tolist()
                            data.loc[(data["Mã HP"] == i) & (data["Mã SV"].isin(s_i)), "Mã HP"] = f"{i}_{s}"

            if solver.ObjectiveValue() == 0:
                logger.info("Solution found with zero objective value.")
                break
        else:
            logger.error("No optimal solution found.")
            break

    


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

    #logger.debug(f"\nĐợt thi: {dot_thi}\nTỉnh thành: {tinh}\nSố slot: {time_slot}\nSố phòng khả dụng: {n_avai_room}\nSố sinh viên 1 phòng {student_per_room}\nSố sinh viên min 1 phòng: {min_sv_hp}\n")

    file_path = f"{input_dir}/{dot_thi}/{dot_thi}_{tinh}.csv"
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
        
        half_day = time_slot//2
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
            logger.info(f"Objective value 1: {solver.value(obj)}")
            for t in range(n_time_slot):
                logger.info(f"Time slot {t}: ")
                logger.info(
                    f"Number of rooms: {sum((solver.value(x[(i, t)]) * hp.loc[i, 'rooms']) for i in hp.index)}"
                )
                for i in hp.index:
                    if solver.value(x[(i, t)]) > 0:
                        data.loc[data["Mã HP"] == i, "Time slot"] = t
                        logger.info(f"Exam {i}")

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
                        and solver.value(same[(i, j)]) > 0
                    ):
                        split = i
                        if len(i.split("_")) < max_division:
                            split = i
                        if len(j.split("_")) < max_division:
                            split = j
                        if hp.loc[i, "sv"] < hp.loc[j, "sv"] or len(i.split("_")) > len(j.split("_")):
                            split = j
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
                    best_y = None
                    best_nc = 0
                    best_score = -1

                    # K-mean 
                    for nc in range(2, min((hp.loc[i, "sv"] // min_hp_sv) + 1, 11)):
                        kmeans = KMeansConstrained(
                            n_clusters=nc, size_min=min_hp_sv, n_jobs=n_jobs
                        )
                        yy = kmeans.fit_predict(StandardScaler().fit_transform(X))
                        score = silhouette_score(X, yy)
                        if score > best_score:
                            best_nc = nc
                            best_score = score
                            best_y = yy
                    
                    logger.info(
                        f"Splitting {i} ({hp.loc[i, 'sv']}) into {best_nc} clusters"
                    )
                    X["cluster"] = best_y

                    for s in range(best_nc):
                        s_i = X.loc[X["cluster"] == s].index.tolist()
                        data.loc[
                            (data["Mã HP"] == i) & (data["Mã SV"].isin(s_i)), "Mã HP"
                        ] = f"{i}_{s}"

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
                solver.parameters.max_time_in_seconds = 90  
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

    data.to_csv(f"{folder_out}/{dot_thi}_{tinh}.csv", index=False, encoding='utf-8-sig')
    data.to_csv(f"{folder_out}/{dot_thi}_{tinh}_raw.csv", index=False, encoding='utf-8-sig')

    def arranging_room(data):
        data["So SV"] = data.groupby("Mã HP")["Mã SV"].transform("count")
        data = data.sort_values(
            by=["Time slot", "Khoa giảng dạy", "Mã HP", "Mã LHP", "So SV", "Tên", "Họ Lót", "Mã SV"],
            ascending=[True, True, True, True, False, True, True, True],
        )

        #n_time_slot = data["Time slot"].nunique()
        for i in range(n_time_slot):
            room_id = 0
            for hp in data.loc[data["Time slot"] == i, "Mã HP"].unique():
                data.loc[(data["Time slot"] == i) & (data["Mã HP"] == hp), "Room"] = range(
                    len(data.loc[(data["Time slot"] == i) & (data["Mã HP"] == hp), "Room"])
                )
                data.loc[(data["Time slot"] == i) & (data["Mã HP"] == hp), "Room"] = (
                    data.loc[(data["Time slot"] == i) & (data["Mã HP"] == hp), "Room"] // student_per_room
                ) + room_id
                room_id = (
                    data.loc[(data["Time slot"] == i) & (data["Mã HP"] == hp), "Room"].max()
                    + 1
                )

        for i in range(n_time_slot):
            rooms = data.loc[data["Time slot"] == i].groupby("Room")["Mã SV"].count()
            rooms = rooms.sort_values()
            #print(rooms)
            while len(rooms) >= 2 and rooms.iloc[0] + rooms.iloc[1] <= student_per_room:
                data.loc[
                    (data["Time slot"] == i) & (data["Room"] == rooms.index[0]), "Room"
                ] = rooms.index[0]
                data.loc[
                    (data["Time slot"] == i) & (data["Room"] == rooms.index[1]), "Room"
                ] = rooms.index[0]
                rooms = data.loc[data["Time slot"] == i].groupby("Room")["Mã SV"].count()
                rooms = rooms.sort_values()

            #print(rooms)
        return data

    data = arranging_room(data)      


    logger.debug(f"\n{data}")

    data = data[["Địa điểm","Mã SV","Họ Lót","Tên","Mã HP","Mã LHP","Tên HP","Số TC", "Bộ môn giảng dạy","Khoa giảng dạy", "Time slot","Room"]]

    data = data.sort_values(
        by=["Time slot", "Room", "Mã HP", "Mã LHP", "Khoa giảng dạy", "Tên", "Họ Lót", "Mã SV"],
        ascending=[True, True, True, True, True, True, True, True],
    )

    #Re-index room num
    all_ts_room = []
    for i in range (n_time_slot):
        ts_room = []
        for room in data.loc[data["Time slot"] == i, "Room"].unique():
            ts_room.append(room)
        all_ts_room.append(ts_room)

    process_ts_room = []
    for i in range (n_time_slot):
        n = len(all_ts_room[i])
        ts_room = []
        for j in range (n):
            ts_room.append(j)
        process_ts_room.append(ts_room)

    for i in range (n_time_slot):
        for j in range (len(all_ts_room[i])):
            data.loc[
                (data["Time slot"] == i)  & (data["Room"] == all_ts_room[i][j]), "Room"
                ] = process_ts_room[i][j]

    logger.debug(f"\n{data}")
    data = data[["Địa điểm","Mã SV","Họ Lót","Tên","Mã HP","Mã LHP","Tên HP","Số TC", "Bộ môn giảng dạy","Khoa giảng dạy", "Time slot","Room"]]
    #data.to_csv(f"{folder_out}/{dot_thi}_{tinh}.csv", encoding  = 'utf-8-sig', index=False, mode='w+')

    data = data.sort_values(
        by=["Time slot", "Room", "Mã HP", "Tên", "Họ Lót", "Mã SV"],
        ascending=[True, True, True, True, True, True],
    )

    logger.debug(f"\n{data}")

    data.to_csv(f"{folder_out}/{dot_thi}_{tinh}.csv", encoding  = 'utf-8-sig', index=False, mode='w+')

if __name__ == "__main__":
    p = psutil.Process()
    p.cpu_affinity([1, 2, 3, 4])
    app.run(main)
    