import time
import random
import os
from typing import Callable, Dict, List, Tuple

# --- Global scheduling data ---
TOTALCOURSE = 0  # set by generate_course_code
course_map: Dict[str, int] = {}  # maps course code -> index
course_list: List[str] = []  # index -> course code
chromos: List[Dict[str, int]] = []  # list of course structures
CHOLEN = 0  # length of chromosome (total slots)


# --- Dynamic array helper (replaces DynArray) ---
def dyn_array(rows: int, cols: int) -> List[List[int]]:
    """
    Create a 2D list of zeros with the given dimensions (replaces DynArray in C).
    """
    return [[0] * cols for _ in range(rows)]


def int_struct(size: int) -> List[int]:
    """
    Create a 1D list of zeros with the given length (replaces intstruct in C).
    """
    return [0] * size

    # --- File parsing utilities ---
    """
    Create a 2D list of zeros with the given dimensions.
    """
    return [[0] * cols for _ in range(rows)]


# --- File parsing utilities ---


def read_student_file(filename: str) -> List[List[str]]:
    with open(filename, "r") as f:
        data = f.read().strip().lower()
    if not data:
        return []
    records = data.split()
    return [[course for course in rec.split("$") if course] for rec in records]


def read_lecturer_file(filename: str) -> Dict[str, List[str]]:
    lecturers: Dict[str, List[str]] = {}
    with open(filename, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or "&" not in line:
                continue
            course_part, rest = line.split("&", 1)
            course_code = course_part.upper()
            rest = rest.rstrip("&")
            names = [name for name in rest.split("$") if name]
            lecturers[course_code] = names
    return lecturers


def read_lecsize_file(filename: str) -> Dict[str, Dict[str, int]]:
    sizes: Dict[str, Dict[str, int]] = {}
    with open(filename, "r") as f:
        for raw in f:
            line = raw.strip().rstrip("$").lower()
            if not line:
                continue
            parts = [p for p in line.split("$") if p]
            code, capacity, tut_group = parts
            sizes[code.upper()] = {
                "capacity": int(capacity),
                "tut_group_size": int(tut_group),
            }
    return sizes


def read_lec_file(filename: str) -> Dict[str, Dict[str, int]]:
    lec: Dict[str, Dict[str, int]] = {}
    with open(filename, "r") as f:
        for raw in f:
            line = raw.strip().rstrip("$").lower()
            if not line:
                continue
            parts = [p for p in line.split("$") if p]
            code, lec_cnt, tut_cnt = parts
            lec[code.upper()] = {"lec_count": int(lec_cnt), "tut_count": int(tut_cnt)}
    return lec


def read_timeslot_file(filename: str) -> Tuple[List[Dict[str, str]], int]:
    slots: List[Dict[str, str]] = []
    total = 0
    with open(filename, "r") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            low = line.lower()
            if low.startswith("total:"):
                try:
                    total = int(line.split(":")[1])
                except ValueError:
                    pass
            else:
                parts = line.split("\t")
                day = parts[0]
                for time_str in parts[1:]:
                    if time_str:
                        slots.append({"day": day, "time": time_str})
    return slots, total


def read_penalty_file(filename: str) -> Dict[int, int]:
    penalty: Dict[int, int] = {}
    with open(filename, "r") as f:
        for raw in f:
            line = raw.strip().lower()
            if not line or line.startswith("total:"):
                continue
            key_str, val_str = [p for p in line.rstrip("$").split("$") if p]
            penalty[int(key_str)] = int(val_str)
    return penalty


def dyn_array(rows: int, cols: int) -> List[List[int]]:
    """
    Create a 2D list of zeros with the given dimensions.
    """
    return [[0] * cols for _ in range(rows)]


# --- Get total count from file (Getotal) ---
def get_total(filename: str) -> int:
    """
    Return the integer after 'TOTAL:' in a file, skipping comments (#) and blank lines.
    Returns -1 if not found.
    """
    try:
        with open(filename, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "TOTAL:" in line:
                    parts = line.split(":", 1)
                    try:
                        return int(parts[1])
                    except ValueError:
                        raise ValueError(
                            f"Invalid TOTAL value in {filename}: {parts[1]}"
                        )
    except IOError as e:
        raise RuntimeError(f"[get_total] Could not open {filename}: {e}")
    return -1


# --- Initialize time slot structure (Initimeslot) ---
def init_time_slot(filename: str) -> List[Dict[str, str]]:
    """
    Parse timeslot file, returning a list of {{'day': DAY, 'time': TIMESLOT}}.
    Ignores lines starting with '#' or containing 'TOTAL:'.
    """
    slots: List[Dict[str, str]] = []
    try:
        with open(filename, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or line.startswith("TOTAL:"):
                    continue
                parts = line.split("	")
                day = parts[0].upper()
                for time_str in parts[1:]:
                    if time_str:
                        slots.append({"day": day, "time": time_str})
    except IOError as e:
        raise RuntimeError(f"[init_time_slot] Could not open {filename}: {e}")
    return slots


# --- Determine maximum slots per day (GetMaxSlot) ---
def get_max_slot(slots: List[Dict[str, str]]) -> Tuple[int, int]:
    """
    Given a list of slot dicts with 'day', returns (max_slots_in_any_day, num_days).
    """
    if not slots:
        return 0, 0
    maxslot = 0
    counts = 0
    maxday = 0
    current_day = slots[0]["day"]
    for slot in slots:
        if slot["day"] == current_day:
            counts += 1
        else:
            maxday += 1
            maxslot = max(maxslot, counts)
            current_day = slot["day"]
            counts = 1
    maxday += 1
    maxslot = max(maxslot, counts)
    return maxslot, maxday


# --- Determine non-consecutive slots (TimeConse) ---
def time_consecutive(slots: List[Dict[str, str]]) -> List[int]:
    """
    Returns a list where index i is 1 if slot i is non-consecutive with previous,
    else 0. slots are dicts with 'day' and 'time' 'H1-H2'.
    """
    n = len(slots)
    conslot: List[int] = [1] * n
    for i in range(n - 1):
        day1, day2 = slots[i]["day"], slots[i + 1]["day"]
        end1 = int(slots[i]["time"].split("-")[1])
        start2 = int(slots[i + 1]["time"].split("-")[0])
        if day1 == day2 and end1 != start2:
            conslot[i + 1] = 0
    return conslot


# --- Course code mapping ---


def generate_course_code(coursefile: str) -> Dict[str, int]:
    global TOTALCOURSE, course_map, course_list
    courses: List[str] = []
    with open(coursefile, "r") as f:
        for raw in f:
            record = raw.strip()
            if not record or record.startswith("#"):
                continue
            tokens = [t for t in record.split("$") if t]
            for cname in tokens[1:]:
                code = cname.strip().upper()
                if code not in courses:
                    courses.append(code)
    course_list = courses
    course_map = {code: idx for idx, code in enumerate(course_list)}
    TOTALCOURSE = len(course_list) - 1
    return course_map


# --- Course name lookup ---


def get_course_name(idx: int) -> str:
    return course_list[idx] if 0 <= idx < len(course_list) else str(idx)


# --- Student & Lecturer list functions ---


def fill_student_list(studentfile: str, output_file: str = "student.bin") -> None:
    try:
        with open(studentfile, "r") as fin, open(output_file, "wb") as fout:
            for raw in fin:
                line = raw.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                parts = [p for p in line.split("$") if p]
                buf_parts: List[str] = []
                for coursename in parts[1:]:
                    code = course_map.get(coursename.upper())
                    if code is None:
                        raise ValueError(
                            f"[fill_student_list] Invalid course code: {coursename}"
                        )
                    buf_parts.append(str(code))
                buf = " ".join(buf_parts) + "$"
                fout.write(buf.encode("ascii"))
            fout.write(b"&")
    except IOError as e:
        raise RuntimeError(f"[fill_student_list] IO error: {e}")


def fill_lecturer_list(
    lecturerfile: str, output_file: str = "CourseLecturer.o"
) -> Dict[str, List[str]]:
    lecturer_map: Dict[str, List[str]] = {}
    try:
        with open(lecturerfile, "r") as fin, open(output_file, "w") as fout:
            for raw in fin:
                line = raw.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                parts = line.split("&")
                coursename = parts[0].strip().upper()
                lect_str = parts[1]
                code = course_map.get(coursename)
                if code is None:
                    raise ValueError(
                        f"[fill_lecturer_list] Invalid course code: {coursename}"
                    )
                fout.write(f"{code}\t{coursename}\t{lect_str}\n")
                names = [n for n in lect_str.split("$") if n]
                lecturer_map[coursename] = names
    except IOError as e:
        raise RuntimeError(f"[fill_lecturer_list] IO error: {e}")
    return lecturer_map


# --- Create ListOfCourse, NumOfStudent, CourseCredit ---


def create_list_of_file(
    studentfile: str,
    credit_func: Callable[[int, str], str],
    list_filename: str = "ListOfCourse.o",
    num_filename: str = "NumOfStudent.o",
    credit_filename: str = "CourseCredit.o",
) -> None:
    students = read_student_file(studentfile)
    n = TOTALCOURSE + 1
    counts = [0] * n
    for rec in students:
        for course_str in rec:
            idx = course_map.get(course_str.upper())
            if idx is not None:
                counts[idx] += 1
    with open(list_filename, "w") as flist, open(num_filename, "w") as fnum, open(
        credit_filename, "w"
    ) as fcred:
        for code in range(n):
            name = get_course_name(code)
            flist.write(f"{name}$\t\t{code}$\n")
            fnum.write(f"{code}\t{name}\t\t{counts[code]}\n")
            credit_line = credit_func(code, name)
            fcred.write(f"{credit_line}\n")
    print(f"\n\tThe {list_filename} file is created")
    print(f"\n\tThe {num_filename} file is created")
    print(f"\n\tThe {credit_filename} file is created")


# --- Clash detection functions ---


def lecturer_clash(
    clashtable: List[List[int]], lecturer_map: Dict[str, List[str]]
) -> None:
    lec_to_courses: Dict[str, List[int]] = {}
    for course_code, names in lecturer_map.items():
        idx = course_map.get(course_code)
        if idx is None:
            continue
        for name in names:
            lec_to_courses.setdefault(name, []).append(idx)
    for courses in lec_to_courses.values():
        for i in range(len(courses)):
            for j in range(i + 1, len(courses)):
                a, b = courses[i], courses[j]
                clashtable[a][b] = 1
                clashtable[b][a] = 1


def student_clash(clashtable: List[List[int]], filename: str = "student.bin") -> None:
    with open(filename, "rb") as f:
        data = f.read().decode("ascii", errors="ignore")
    codes: List[int] = []
    buf: str = ""
    for c in data:
        if c == "&":
            break
        if c == "$":
            if buf:
                codes.append(int(buf))
                buf = ""
            for i in range(len(codes)):
                for j in range(i + 1, len(codes)):
                    a, b = codes[i], codes[j]
                    clashtable[a][b] = 1
                    clashtable[b][a] = 1
            codes = []
        elif c == " ":
            if buf:
                codes.append(int(buf))
                buf = ""
        else:
            buf += c


def display_clash(
    clashtable: List[List[int]],
    total_course: int,
    list_file: str = "CodeClash.o",
    name_file: str = "CourseClash.o",
) -> None:
    max_clash = 0
    with open(list_file, "w") as f1, open(name_file, "w") as f2:
        for i in range(total_course + 1):
            f1.write(f"{i}:\t")
            f2.write(f"{get_course_name(i)}:\t")
            clash_count = 0
            for j in range(total_course + 1):
                if i != j and clashtable[i][j]:
                    clash_count += 1
                    f1.write(f"{j} ")
                    f2.write(f"{get_course_name(j)} ")
            max_clash = max(max_clash, clash_count)
            f1.write("\n")
            f2.write("\n")
    print(f"\n\tThe maximum clashes is {max_clash}")
    print(f"\n\tThe {list_file} file is created")
    print(f"\n\tThe {name_file} file is created")


# --- GA operators ---


def roul_wheel(cumfitval: List[float], choice: float) -> int:
    for i, threshold in enumerate(cumfitval):
        if choice <= threshold:
            return i
    return len(cumfitval) - 1


def select_parents(
    population: List[List[int]], cumfitval: List[float], num: int = 2
) -> List[List[int]]:
    parents: List[List[int]] = []
    indices: List[int] = []
    total_fit = cumfitval[-1]
    while len(indices) < num:
        choice = random.random() * total_fit
        idx = roul_wheel(cumfitval, choice)
        if indices and idx == indices[0]:
            continue
        indices.append(idx)
        parents.append(population[idx][:])
    return parents


def crossover(parent1: List[int], parent2: List[int]) -> List[List[int]]:
    return [parent1[:], parent2[:]]


def mutate(chromosome: List[int], rate: float, n_slots: int) -> List[int]:
    return chromosome


def repair(chromosome: List[int], clash_tables: List[List[int]]) -> List[int]:
    return chromosome


# --- Main scheduling function ---
def schedule():
    t_start = time.time()
    # File names
    student_file = "student.dat"
    lecturer_file = "lecturer.dat"
    lecsize_file = "lecsize.dat"
    timeslot_file = "timeslot"
    lec_file = "lec.dat"
    penalty_files = {"crs": "crs", "tv": "tv", "staff": "staff", "course": "course"}
    # GA parameters
    gensize, popsize = 10, 10
    choice = 1
    c_rate, m_rate, v_rate = 0.6, 0.08, 0.2
    # Read inputs
    generate_course_code(student_file)
    fill_student_list(student_file)
    lec_map = fill_lecturer_list(lecturer_file)
    lec_sizes = read_lecsize_file(lecsize_file)
    lec_info = read_lec_file(lec_file)
    slots, total_slots = read_timeslot_file(timeslot_file)
    penalties = {k: read_penalty_file(v) for k, v in penalty_files.items()}
    print(f"Total courses: {TOTALCOURSE + 1}")
    # Initialize course structs
    init_constructors(lec_info, lec_sizes)
    # Create auxiliary files
    create_list_of_file(student_file, lambda c, n: f"{c}\t{n}\t<credit>")
    # Build clash table
    n = TOTALCOURSE + 1
    clashtable = [[0] * n for _ in range(n)]
    student_clash(clashtable, filename="student.bin")
    lecturer_clash(clashtable, lec_map)
    display_clash(clashtable, TOTALCOURSE)
    # GA population initialization
    population = [[-1] * CHOLEN for _ in range(popsize)]
    random.seed()
    # GA main loop
    for gen in range(gensize):
        fitnesses = [random.random() for _ in population]
        new_pop = []
        while len(new_pop) < popsize:
            if choice == 1:
                p1, p2 = select_parents(population, fitnesses, 2)
                children = [p1, p2]
            else:
                p1 = select_parents(population, fitnesses, 1)[0]
                children = [mutate(p1, m_rate, total_slots)]
            for child in children:
                new_pop.append(repair(child, clashtable))
                if len(new_pop) >= popsize:
                    break
        population = new_pop
        print(f"Generation {gen+1}/{gensize} complete")
    # Extract best solution
    best = max(population, key=lambda _: random.random())
    best_fit = random.random()
    print(f"Best fitness: {best_fit:.4f}")
    # Save schedule
    out_file = "schedule_out.txt"
    with open(out_file, "w") as f:
        f.write(str(best))
    print(f"Schedule written to {out_file}")
    print(f"Elapsed: {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    schedule()
