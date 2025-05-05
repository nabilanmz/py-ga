#!/usr/bin/env python3
import time
import random
import os
from typing import List, Dict, Tuple, Callable


# =============================================================================
#  DYNAMIC ARRAY & INT STRUCT HELPERS (DynArray / intstruct)
# =============================================================================
def dyn_array(rows: int, cols: int) -> List[List[int]]:
    """
    Create a 2D list of zeros with the given dimensions.
    """
    return [[0] * cols for _ in range(rows)]


def int_struct(size: int) -> List[int]:
    """
    Create a 1D list of zeros with the given length.
    """
    return [0] * size


# =============================================================================
#  FILE PARSING UTILITIES
# =============================================================================
#  read_student_file: parse student.dat


def read_student_file(filename: str) -> List[List[str]]:
    with open(filename, "r") as f:
        data = f.read().strip().lower()
    if not data:
        return []
    records = data.split()
    return [[course for course in rec.split("$") if course] for rec in records]


#  read_lecturer_file: parse lecturer.dat


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


#  read_lecsize_file: parse lecsize.dat


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


#  read_lec_file: parse lec.dat


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


#  read_timeslot_file: parse timeslot, also return TOTAL


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
                day = parts[0].upper()
                for time_str in parts[1:]:
                    if time_str:
                        slots.append({"day": day, "time": time_str})
    return slots, total


#  read_penalty_file: parse penalty files


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


# =============================================================================
#  GETOTAL: retrieve integer after "TOTAL:" in file
# =============================================================================
def get_total(filename: str) -> int:
    try:
        with open(filename, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "TOTAL:" in line:
                    parts = line.split(":", 1)
                    return int(parts[1])
    except IOError as e:
        raise RuntimeError(f"[get_total] Could not open {filename}: {e}")
    return -1


# =============================================================================
#  INITIMESLOT: initialize time slots from file
# =============================================================================
def init_time_slot(filename: str) -> List[Dict[str, str]]:
    slots: List[Dict[str, str]] = []
    try:
        with open(filename, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or line.startswith("TOTAL:"):
                    continue
                parts = line.split("\t")
                day = parts[0].upper()
                for time_str in parts[1:]:
                    if time_str:
                        slots.append({"day": day, "time": time_str})
    except IOError as e:
        raise RuntimeError(f"[init_time_slot] Could not open {filename}: {e}")
    return slots


# =============================================================================
#  GETMAXSLOT: determine max slots per day and number of days
# =============================================================================
def get_max_slot(slots: List[Dict[str, str]]) -> Tuple[int, int]:
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


# =============================================================================
#  TIMECONSE: mark non-consecutive time slots
# =============================================================================
def time_consecutive(slots: List[Dict[str, str]]) -> List[int]:
    n = len(slots)
    conslot: List[int] = [1] * n
    for i in range(n - 1):
        day1 = slots[i]["day"]
        day2 = slots[i + 1]["day"]
        end1 = int(slots[i]["time"].split("-")[1])
        start2 = int(slots[i + 1]["time"].split("-")[0])
        if day1 == day2 and end1 != start2:
            conslot[i + 1] = 0
    return conslot


# =============================================================================
#  GLOBAL DATA & GENERATE_COURSE_CODE
# =============================================================================
TOTALCOURSE = 0
course_map: Dict[str, int] = {}
course_list: List[str] = []
chromos: List[Dict[str, int]] = []
CHOLEN = 0


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


# =============================================================================
#  GET_COURSENAME: name lookup by index
# =============================================================================
def get_course_name(idx: int) -> str:
    return course_list[idx] if 0 <= idx < len(course_list) else str(idx)


# =============================================================================
#  IniCostruct: initialize chromos from lec/tut info
# =============================================================================
def init_constructors(
    lec_info: Dict[str, Dict[str, int]], lec_sizes: Dict[str, Dict[str, int]]
) -> None:
    global chromos, CHOLEN
    n = TOTALCOURSE + 1
    chromos = [{"lec": 0, "tut": 0, "from": 0, "to": 0} for _ in range(n)]
    for code, data in lec_info.items():
        idx = course_map.get(code)
        if idx is None:
            raise ValueError(f"[init_constructors] Invalid course code: {code}")
        chromos[idx]["lec"] = data["lec_count"]
        chromos[idx]["tut"] = data["tut_count"]
    for i in range(n):
        if i == 0:
            chromos[i]["from"] = 0
        else:
            chromos[i]["from"] = chromos[i - 1]["to"] + 1
        chromos[i]["to"] = (
            chromos[i]["from"] + chromos[i]["lec"] + chromos[i]["tut"] - 1
        )
    CHOLEN = chromos[TOTALCOURSE]["to"] + 1


# =============================================================================
#  filllecturerlist: write CourseLecturer.o and build lecturer_map
# =============================================================================
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


# =============================================================================
#  fillstudentlist: write student.bin from student.dat
# =============================================================================
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


# =============================================================================
#  Create_List_Of_File: ListOfCourse.o, NumOfStudent.o, CourseCredit.o
# =============================================================================
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
            fcred.write(f"{credit_func(code, name)}\n")
    print(f"\n\tThe {list_filename} file is created")
    print(f"\n\tThe {num_filename} file is created")
    print(f"\n\tThe {credit_filename} file is created")


# =============================================================================
#  LecturerClash: record clashes based on same lecturer
# =============================================================================
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


# =============================================================================
#  StudentClash: record clashes based on same student
# =============================================================================
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


# =============================================================================
#  Display_clash: write CodeClash.o and CourseClash.o
# =============================================================================
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


# =============================================================================
#  GA OPERATORS: RoulWheel, ParentSelect, Crossover, Mutation, Repair
# =============================================================================
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


# =============================================================================
#  IniAschCos: initialize chromosomes with already scheduled courses
# =============================================================================
# =============================================================================
#  CONVERT PARTIAL TIME STRING TO 'DAY TIME' (partialconvert)
# =============================================================================
def partial_convert(timeptr: str) -> str:
    """
    Convert a compact time record into standardized 'DAY TIME' format.
    E.g. 'Mon 8-9'→'MON 8-9'. Splits on first space: day to uppercase, time unchanged.
    """
    parts = timeptr.strip().split(" ", 1)
    if len(parts) == 2:
        day, times = parts
        return f"{day.upper()} {times}"
    return timeptr.upper()


# =============================================================================
#  FIND TIME SLOT INDEX FOR A 'DAY TIME' STRING (Matchtimeslot)
# =============================================================================
def match_time_slot(str_times: str, slots: List[Dict[str, str]]) -> int:
    """
    Given 'DAY TIME', search slots list for matching entry and return its index.
    Returns -1 if not found.
    """
    target = str_times.strip().upper()
    for i, slot in enumerate(slots):
        candidate = f"{slot['day']} {slot['time']}".upper()
        if candidate.startswith(target):
            return i
    return -1


# =============================================================================
#  INITIALIZE COURSE TIMESLOT (Initchorec)
# =============================================================================
def init_chorec(
    cos: List[Dict[str, int]],
    slots: List[Dict[str, str]],
    coursecode: int,
    rec: str,
    cho: List[List[int]],
    nslot: int,
    popsize: int,
    lecture: bool,
) -> None:
    """
    Initialize already scheduled times for a course component.
    rec: string of times separated by '$'
    lecture: False for lecture, True for tutorial
    """
    # Determine index range for this course component
    if lecture:
        start = cos[coursecode]["from"] + cos[coursecode]["lec"]
        length = cos[coursecode]["tut"]
    else:
        start = cos[coursecode]["from"]
        length = cos[coursecode]["lec"]
    end = start + length - 1

    # Split the record string into individual time tokens
    times = [t for t in rec.split("$") if t]
    for t in times:
        # Convert to standardized time string, e.g. '8-9'
        timestr = partial_convert(
            t
        )  # You should implement partial_convert to match C partialconvert
        if start > end:
            raise ValueError("[init_chorec] Invalid range!")
        # Find the corresponding slot index
        timecode = match_time_slot(
            timestr, slots
        )  # Implement match_time_slot to mirror C Matchtimeslot
        if timecode == -1:
            raise ValueError("[init_chorec] Invalid time slot!")
        # Assign this timecode to all chromosomes at the 'start' gene position
        for row in range(popsize):
            cho[row][start] = timecode
        start += 1


def init_asch_cos(
    scheslotfile: str,
    cos: List[Dict[str, int]],
    slots: List[Dict[str, str]],
    cho: List[List[int]],
    nslot: int,
    popsize: int,
) -> None:
    try:
        with open(scheslotfile, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("&")
                coursename = parts[0].upper()
                lec, tut = parts[1], parts[2]
                code = course_map.get(coursename)
                if code is None:
                    raise ValueError(
                        f"[init_asch_cos] Invalid course code: {coursename}"
                    )
                if lec != "X":
                    init_chorec(
                        cos, slots, code, lec, cho, nslot, popsize, lecture=True
                    )
                if tut != "X":
                    init_chorec(
                        cos, slots, code, tut, cho, nslot, popsize, lecture=False
                    )
    except IOError as e:
        raise RuntimeError(f"[init_asch_cos] Could not open {scheslotfile}: {e}")


# =============================================================================
#  InitChos: initialize chromosomes randomly with constraints
# =============================================================================
def init_chos(
    cos: List[Dict[str, int]],
    clashtable: List[List[int]],
    cho: List[List[int]],
    fixslot: List[int],
    popsize: int,
    nslot: int,
) -> None:
    for row in range(popsize):
        for col in range(len(cho[row])):
            cho[row][col] = -1
    random.seed()
    for row in range(popsize):
        for col in range(len(cho[row])):
            cho[row][col] = random.randrange(nslot)


# =============================================================================
#  Backtracking: try swapping slots to resolve clashes
# =============================================================================
def backtracking(
    clashtable: List[List[int]],
    cos: List[Dict[str, int]],
    cho: List[int],
    Fixslot: List[int],
    code: int,
    col: int,
    nslot: int,
) -> bool:
    """
    Attempt to resolve clashes by swapping current gene at col with another slot.
    Returns True if successful, False otherwise.
    """
    # initialize helper arrays
    timeslot = int_struct(nslot)
    freeslot = int_struct(nslot)
    tmpcho = list(cho)  # copy
    tmpslot = int_struct(nslot)

    # count clashes per timeslot from other courses
    for cpos in range(TOTALCOURSE + 1):
        if cpos == code:
            continue
        if clashtable[cpos][code]:
            for gpos in range(cos[cpos]["from"], cos[cpos]["to"] + 1):
                sl = tmpcho[gpos]
                if sl == -1:
                    continue
                # skip if both are tutorials and beyond lecture range
                if (
                    col > cos[code]["from"] + cos[code]["lec"] - 1
                    and gpos > cos[cpos]["from"] + cos[cpos]["lec"] - 1
                ):
                    continue
                timeslot[sl] += 1
    # count self-clashes
    for gpos in range(cos[code]["from"], cos[code]["to"] + 1):
        sl = tmpcho[gpos]
        if sl == -1:
            continue
        timeslot[sl] += 1

    NumofClash = 1
    NumofSlot = 0
    # attempt increasing clash levels
    while NumofSlot < nslot:
        # collect slots with exactly NumofClash clashes
        tpos = 0
        for pos in range(nslot):
            if timeslot[pos] == NumofClash:
                tmpslot[tpos] = pos
                tpos += 1
        # try each candidate
        while tpos:
            randslot = random.randrange(tpos)
            cntl = True
            npos = 0
            # for each gene, if matches candidate, try swap
            for gpos in range(CHOLEN):
                if tmpcho[gpos] != tmpslot[randslot]:
                    continue
                # find alternative free slots for this gpos
                frees = get_free_slots(clashtable, cos, tmpcho, code, gpos, nslot)
                if not frees:
                    cntl = False
                    break
                # perform swap
                tmpcho[col], tmpcho[gpos] = tmpcho[gpos], random.choice(frees)
                npos += 1
                if npos >= NumofClash:
                    break
            if cntl and npos:
                # commit
                cho[:] = tmpcho
                return True
            # remove this candidate and continue
            tmpslot[randslot : tpos - 1] = tmpslot[randslot + 1 : tpos]
            tpos -= 1
            NumofSlot += NumofClash
        NumofClash += 1
    return False


# =============================================================================
#  getfreeslot: detect free slots for a course item
# =============================================================================
# =============================================================================
def get_free_slots(
    clashtable: List[List[int]],
    cos: List[Dict[str, int]],
    cho: List[int],
    code: int,
    col: int,
    nslot: int,
) -> List[int]:
    clashslot = [0] * nslot
    for cpos in range(TOTALCOURSE + 1):
        if cpos == code:
            continue
        if clashtable[cpos][code]:
            for gpos in range(cos[cpos]["from"], cos[cpos]["to"] + 1):
                if (
                    col > cos[code]["from"] + cos[code]["lec"] - 1
                    and gpos > cos[cpos]["from"] + cos[cpos]["lec"] - 1
                ):
                    break
                slot = cho[gpos]
                if slot >= 0:
                    clashslot[slot] = 1
    for gpos in range(cos[code]["from"], cos[code]["to"] + 1):
        slot = cho[gpos]
        if slot == -1 and gpos == col:
            continue
        if slot >= 0:
            clashslot[slot] = 1
    freeslot = [tpos for tpos in range(nslot) if clashslot[tpos] == 0]
    print("sasalskalsalsla")
    return freeslot


# =============================================================================
#  Main scheduling function
# =============================================================================
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

    # Read and prepare data
    generate_course_code(student_file)
    print(f"Total courses: {TOTALCOURSE + 1}")
    fill_student_list(student_file)
    lecturer_map = fill_lecturer_list(lecturer_file)
    lec_sizes = read_lecsize_file(lecsize_file)
    lec_info = read_lec_file(lec_file)
    slots, total_slots = read_timeslot_file(timeslot_file)
    penalties = {k: read_penalty_file(v) for k, v in penalty_files.items()}
    init_constructors(lec_info, lec_sizes)

    # Create auxiliary files
    create_list_of_file(student_file, lambda c, n: f"{c}\t{n}\t<credit>")

    # Build clash table
    n = TOTALCOURSE + 1
    clashtable = dyn_array(n, n)
    student_clash(clashtable, filename="student.bin")
    lecturer_clash(clashtable, lecturer_map)
    display_clash(clashtable, TOTALCOURSE)

    # Initialize population
    cho = dyn_array(popsize, CHOLEN)
    fixslot = int_struct(CHOLEN)
    init_asch_cos("", chromos, slots, cho, total_slots, popsize)
    init_chos(chromos, clashtable, cho, fixslot, popsize, total_slots)

    # GA main loop (simplified)
    population = cho
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

    # Finalize
    best = max(population, key=lambda _: random.random())
    print(f"Best individual: {best}")
    print(f"Elapsed: {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    schedule()
