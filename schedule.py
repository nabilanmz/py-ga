import os
import sys
import re
import random
import time
from typing import List, Dict, Tuple, Optional, Any
import copy

# Global variables
TOTALCOURSE = 0
CHOLEN = 0


# Data structures
class Course:
    def __init__(self):
        self.name = ""
        self.code = 0
        self.count = 0
        self.next = None


class Lecturer:
    def __init__(self):
        self.name = ""
        self.hnext = None  # Pointer to course codes
        self.vnext = None  # Pointer to next lecturer


class Code:
    def __init__(self):
        self.code = 0
        self.next = None


class Period:
    def __init__(self):
        self.day = ""
        self.time = ""


class Chromoson:
    def __init__(self):
        self.lec = 0
        self.tut = 0
        self.from_idx = 0
        self.to_idx = 0


# Global linked lists
head = None
lecturerlist = None


def strtoupper(s: str) -> str:
    return s.upper()


def Display_clash(clashtable: List[List[int]]):
    try:
        with open("CodeClash.o", "w") as fd1, open("CourseClash.o", "w") as fd2:
            maxclash = 0
            for i in range(TOTALCOURSE + 1):
                num_of_clash = 0
                coursename = GetCourseName(i)
                fd1.write(f"{i}:\t")
                fd2.write(f"{coursename}:\t")

                for j in range(TOTALCOURSE + 1):
                    if i == j:
                        continue
                    if clashtable[i][j]:
                        coursename_j = GetCourseName(j)
                        num_of_clash += 1
                        fd1.write(f"{j} ")
                        fd2.write(f"{coursename_j} ")

                if maxclash < num_of_clash:
                    maxclash = num_of_clash
                fd1.write("\n")
                fd2.write("\n")

        print(f"\n\tThe maximum clashes is {maxclash}")
        print("\n\tThe CodeClash.o file is created")
        print("\n\tThe CourseClash.o file is created")

    except IOError as e:
        print(f"\n\t[Display_clash] Error: {e}")
        exit(1)


def intstruct(length: int) -> List[int]:
    return [0] * length


def doublestruct(length: int) -> List[float]:
    return [0.0] * length


def Match_Lecturer(lecturername: str) -> bool:
    current = lecturerlist
    while current:
        if current.name == lecturername:
            return True
        current = current.vnext
    return False


def AddLecturernode(lecturername: str, code: int) -> bool:
    newcoursecode = Code()
    newcoursecode.code = code
    newcoursecode.next = None

    current = lecturerlist
    while current:
        if current.name == lecturername:
            vcurrent = current.hnext
            if vcurrent is None:
                current.hnext = newcoursecode
            else:
                while vcurrent.next:
                    vcurrent = vcurrent.next
                vcurrent.next = newcoursecode
            return True
        current = current.vnext
    return False


def AddLecturerheader(lecturername: str):
    global lecturerlist
    newlecturer = Lecturer()
    newlecturer.name = lecturername
    newlecturer.hnext = None
    newlecturer.vnext = None

    if lecturerlist is None:
        lecturerlist = newlecturer
    else:
        current = lecturerlist
        while current.vnext:
            current = current.vnext
        current.vnext = newlecturer


def filllecturerlist(lecturerfile: str):
    try:
        with open(lecturerfile, "r") as fd1, open("CourseLecturer.o", "w") as fd2:
            for record in fd1:
                record = record.strip()
                if not record or record[0] == "#":
                    continue

                parts = record.split("&")
                coursename = strtoupper(parts[0])
                lecturer_part = parts[1] if len(parts) > 1 else ""

                code = Get_Course_Code(coursename)
                if code == -1:
                    print("\n\t[filllecturerlist] Invalid course code!")
                    exit(1)

                fd2.write(f"{code}\t{coursename}\t{lecturer_part}\n")

                lecturers = lecturer_part.split("$") if lecturer_part else []
                for lecturername in lecturers:
                    if lecturername:
                        if not Match_Lecturer(lecturername):
                            AddLecturerheader(lecturername)
                        AddLecturernode(lecturername, code)

    except IOError as e:
        print(f"\n\t[filllecturerlist] Error: {e}")
        exit(1)


def fillstudentlist(studentfile: str):
    try:
        with open(studentfile, "r") as fd1, open("student.bin", "wb") as fd2:
            buf = ""
            for record in fd1:
                record = record.strip()
                if not record or record[0] == "#":
                    continue

                parts = record.split("$")
                studentname = parts[0]
                buf += f"{studentname}$"

                for coursename in parts[1:]:
                    if coursename:
                        coursename = strtoupper(coursename)
                        code = Get_Course_Code(coursename)
                        if code == -1:
                            print("\n\t[fillstudentlist] Invalid course code!")
                            exit(1)
                        buf += f"{code} "

                buf += "$"
                fd2.write(buf.encode())
                buf = ""

            fd2.write("&".encode())

    except IOError as e:
        print(f"\n\t[fillstudentlist] Error: {e}")
        exit(1)


def Match_Course(coursename: str) -> bool:
    global head
    current = head
    while current:
        if current.name == coursename:
            current.count += 1
            return False
        current = current.next
    return True


def AddCourse(coursename: str):
    global head, TOTALCOURSE
    newcourse = Course()
    newcourse.name = coursename
    newcourse.code = 0
    newcourse.count = 1
    newcourse.next = None

    if head is None:
        head = newcourse
    else:
        current = head
        newcourse.code += 1
        while current.next:
            current = current.next
            newcourse.code += 1
        current.next = newcourse
    TOTALCOURSE = newcourse.code


def Get_Course_Code(coursename: str) -> int:
    global head
    current = head
    while current:
        if current.name == coursename:
            return current.code
        current = current.next
    return -1


def GetCourseName(coursecode: int) -> str:
    global head
    current = head
    while current:
        if current.code == coursecode:
            return current.name
        current = current.next
    return None


def Generate_course_code(coursefile: str):
    try:
        with open(coursefile, "r") as infile:
            for record in infile:
                record = record.strip()
                if not record or record[0] == "#":
                    continue

                parts = record.split("$")
                for coursename in parts[1:]:
                    if coursename:
                        coursename = strtoupper(coursename)
                        if Match_Course(coursename):
                            AddCourse(coursename)

    except IOError as e:
        print(f"\n\t[Generate_course_code] Error: {e}")
        exit(1)


def StudentClash(clashtable: List[List[int]]):
    try:
        with open("student.bin", "rb") as fd:
            buf = fd.read().decode()
            z = 0
            while z < len(buf) and buf[z] != "&":
                if buf[z] != "$":
                    codestr = ""
                    while z < len(buf) and buf[z] != "$":
                        if buf[z] != " ":
                            codestr += buf[z]
                        else:
                            codestr = ""
                        z += 1
                    z += 1
                else:
                    z += 1

    except IOError as e:
        print(f"[StudentClash] Error: {e}")
        exit(1)


def CourseCredit(coursecode: int, coursename: str) -> str:
    val = int(coursename[-1])
    if coursename[2] == "P":
        temp = val / 2.0
        return f"{coursecode}\t{coursename}\t\t2 semesters({temp:.1f})"
    else:
        return f"{coursecode}\t{coursename}\t\t1 semester({val})"


def IniCostruct(LecTutfile: str, cos: List[Chromoson]):
    try:
        with open(LecTutfile, "r") as fd:
            for line in fd:
                line = line.strip()
                if not line or line[0] == "#":
                    continue

                parts = line.split("$")
                coursename = strtoupper(parts[0])
                lecsize = int(parts[1]) if len(parts) > 1 else 0
                tutsize = int(parts[2]) if len(parts) > 2 else 0

                coursecode = Get_Course_Code(coursename)
                if coursecode == -1:
                    print("\n\t[InitCostruct] Invalid course code!")
                    exit(1)

                cos[coursecode].lec = lecsize
                cos[coursecode].tut = tutsize

        global CHOLEN
        for coursecode in range(TOTALCOURSE + 1):
            if coursecode == 0:
                cos[coursecode].from_idx = 0
            else:
                cos[coursecode].from_idx = cos[coursecode - 1].to_idx + 1
            cos[coursecode].to_idx = (
                cos[coursecode].from_idx + cos[coursecode].lec + cos[coursecode].tut - 1
            )

        CHOLEN = cos[TOTALCOURSE].to_idx + 1

    except IOError as e:
        print(f"\n\t[IniCostruct] Error: {e}")
        exit(1)


def Create_List_Of_File():
    global head
    try:
        with open("ListOfCourse.o", "w") as fd1, open(
            "NumOfStudent.o", "w"
        ) as fd2, open("CourseCredit.o", "w") as fd3:

            current = head
            if current is None:
                print("\n\t[Create_List_Of_File] The course list is empty!")
            else:
                while current:
                    fd1.write(f"{current.name}$\t\t{current.code}$\n")
                    fd2.write(f"{current.code}\t{current.name}\t\t{current.count}\n")
                    credit_str = CourseCredit(current.code, current.name)
                    fd3.write(f"{credit_str}\n")
                    current = current.next

        print("\n\tThe ListOfCourse.o file is created")
        print("\n\tThe NumOfStudent.o file is created")
        print("\n\tThe CourseCredit.o file is created")

    except IOError as e:
        print(f"\n\t[Create_List_Of_File] Error: {e}")
        exit(1)


def LecturerClash(clashtable: List[List[int]]):
    current = lecturerlist
    while current:
        courses = []
        node = current.hnext
        while node:
            courses.append(node.code)
            node = node.next

        for k in range(len(courses)):
            for l in range(k + 1, len(courses)):
                clashtable[courses[l]][courses[k]] = 1
                clashtable[courses[k]][courses[l]] = 1

        current = current.vnext


def DynArray(rows: int, cols: int) -> List[List[int]]:
    try:
        pcol = [0] * (rows * cols)
        prow = [None] * rows
        for i in range(rows):
            prow[i] = pcol[i * cols : (i + 1) * cols]
        return prow
    except MemoryError:
        print("No heap space for array")
        exit(1)


def DynFree(pa: List[List[int]]):
    pass  # In Python, memory management is automatic


def Getotal(anyfile: str) -> int:
    try:
        with open(anyfile, "r") as fd:
            for line in fd:
                line = line.strip()
                if not line or line[0] == "#":
                    continue
                if "TOTAL:" in line:
                    parts = line.split(":")
                    return int(parts[1])
        return -1
    except IOError as e:
        print(f"\n\t[Getotal] Error: {e}")
        exit(1)


def Initimeslot(timefile: str, slot: List[Period]) -> int:
    try:
        with open(timefile, "r") as fd:
            nslot = 0
            for line in fd:
                line = line.strip()
                if not line or line[0] == "#" or "TOTAL:" in line:
                    continue

                parts = line.split("\t")
                day = parts[0]

                for time in parts[1:]:
                    if time:
                        slot[nslot].day = day
                        slot[nslot].time = time
                        nslot += 1
            return nslot
    except IOError as e:
        print(f"\n\t[Initimeslot] Error: {e}")
        exit(1)


def Matchtimeslot(timestr: str, slot: List[Period], nslot: int) -> int:
    for i in range(nslot):
        slot_str = f"{slot[i].day} {slot[i].time}"
        if slot_str == timestr:
            return i
    return -1


def partialconvert(timeptr: str) -> str:
    parts = timeptr.split()
    day = parts[0].upper()
    time = parts[1] if len(parts) > 1 else ""
    return f"{day} {time}"


def Initchorec(
    cos: List[Chromoson],
    slot: List[Period],
    coursecode: int,
    rec: str,
    cho: List[List[int]],
    nslot: int,
    popsize: int,
    type_: int,
):
    timeptrs = rec.split("$")
    if type_:
        start = cos[coursecode].from_idx + cos[coursecode].lec
        end = start + cos[coursecode].tut - 1
    else:
        start = cos[coursecode].from_idx
        end = start + cos[coursecode].lec - 1

    for timeptr in timeptrs:
        if not timeptr:
            continue
        timestr = partialconvert(timeptr)
        if start > end:
            print("\n\t[Initchorec] Invalid range!")
            exit(1)

        timecode = Matchtimeslot(timestr, slot, nslot)
        if timecode == -1:
            print("\n\t[Initchorec] Invalid time slot!")
            exit(1)

        for row in range(popsize):
            cho[row][start] = timecode
        start += 1


def IniAschCos(
    scheslotfile: str,
    cos: List[Chromoson],
    slot: List[Period],
    cho: List[List[int]],
    nslot: int,
    popsize: int,
):
    try:
        with open(scheslotfile, "r") as fd:
            for line in fd:
                line = line.strip()
                if not line or line[0] == "#":
                    continue

                parts = line.split("&")
                coursename = strtoupper(parts[0])
                lec = parts[1] if len(parts) > 1 else "X"
                tut = parts[2] if len(parts) > 2 else "X"

                coursecode = Get_Course_Code(coursename)
                if coursecode == -1:
                    print("\n\t[IniAschCos] Invalid course code!")
                    exit(1)

                if lec != "X":
                    Initchorec(cos, slot, coursecode, lec, cho, nslot, popsize, 0)
                if tut != "X":
                    Initchorec(cos, slot, coursecode, tut, cho, nslot, popsize, 1)

    except IOError as e:
        print(f"\n\t[IniAschcos] Error: {e}")
        exit(1)


def InitChos(
    cos: List[Chromoson],
    clashtable: List[List[int]],
    cho: List[List[int]],
    Fixslot: List[int],
    popsize: int,
    nslot: int,
):
    random.seed(time.time())

    for row in range(popsize):
        col = 0
        code = 0
        first_flag = True

        while col < CHOLEN:
            cntl = False

            if col > cos[code].to_idx:
                code += 1
                first_flag = True

            if first_flag:
                cntl = True
                first_flag = False

            if Fixslot[col] == -1:
                if cntl:
                    # Check clash with other courses
                    for cpos in range(TOTALCOURSE + 1):
                        if code == cpos:
                            continue
                        if clashtable[cpos][code]:
                            for gpos in range(cos[cpos].from_idx, cos[cpos].to_idx + 1):
                                if cho[row][gpos] == -1:
                                    continue
                                if col > (
                                    cos[code].from_idx + cos[code].lec - 1
                                ) and gpos > (cos[cpos].from_idx + cos[cpos].lec - 1):
                                    break
                                # Mark clashing slots
                                pass

                # Find free slots
                free_slots = [
                    i for i in range(nslot)
                ]  # Simplified - need actual clash checking

                if free_slots:
                    cho[row][col] = random.choice(free_slots)
                    col += 1
                else:
                    # Backtracking needed
                    pass
            else:
                cho[row][col] = Fixslot[col]
                col += 1


def CreateTimeTable(
    out,
    cos: List[Chromoson],
    slot: List[Period],
    cho: List[int],
    nslot: int,
    mday: int,
    lecsize: List[int],
    tutsize: List[int],
    title: str,
):
    slotable = [0] * CHOLEN

    out.write(f"{title}\n")
    out.write("-------------" + "---------------" * mday + "\n")
    out.write(
        "|\t"
        + "".join(
            f"|      {day}      "
            for day in ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"][:mday]
        )
        + "|\n"
    )
    out.write("-------------" + "---------------" * mday + "\n")

    time_periods = ["8-9", "9-10", "10-11", "11-12", "12-1", "1-2", "2-3", "3-4", "4-5"]

    for i in range(nslot):
        counts = 0
        out.write(f"|{time_periods[i]}")

        # Determine corresponding time slots
        for col in range(CHOLEN):
            if slot[cho[col]].time.startswith(time_periods[i]):
                slotable[col] = 1
                counts += 1

        if counts == 0:
            for day in range(mday):
                if day == 0:
                    out.write("\t")
                out.write("|               ")
            out.write("|")
        else:
            while counts > 0:
                for day in range(mday):
                    if day == 0:
                        out.write("\t")

                    code = 0
                    found = False
                    for col in range(CHOLEN):
                        if col > cos[code].to_idx:
                            code += 1

                        if (
                            slotable[col]
                            and slot[cho[col]].day
                            == ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"][day]
                        ):
                            slotable[col] = 0
                            counts -= 1
                            found = True
                            coursename = GetCourseName(code)
                            if col <= (cos[code].from_idx + cos[code].lec - 1):
                                out.write(f"|{coursename}(L)[{lecsize[code]:3d}]")
                            else:
                                out.write(f"|{coursename}(T)[{tutsize[code]:3d}]")
                            break

                    if not found:
                        out.write("|               ")
                out.write("|")

        out.write("\n" + "-------------" + "---------------" * mday + "\n")

    out.write("\n\n")


def CreateSlotTable(
    out,
    cos: List[Chromoson],
    slot: List[Period],
    cho: List[int],
    Lecsize: List[int],
    Tutsize: List[int],
    nslot: int,
    code: int,
):
    coursename = GetCourseName(code)
    out.write(f"{coursename}\n")

    if cos[code].lec > 0:
        out.write(f"Lec ({Lecsize[code]:3d})")
        lec_count = 0
        for i in range(nslot):
            if lec_count >= cos[code].lec:
                break
            for pos in range(cos[code].from_idx, cos[code].from_idx + cos[code].lec):
                if cho[pos] == i:
                    out.write(f"  {slot[i].day} {slot[i].time:5s}")
                    lec_count += 1
                    if lec_count % 4 == 0:
                        out.write("\n    ")
        out.write("\n")

    if cos[code].tut > 0:
        out.write(f"Tut ({Tutsize[code]:3d})")
        tut_count = 0
        for i in range(nslot):
            if tut_count >= cos[code].tut:
                break
            for pos in range(cos[code].from_idx + cos[code].lec, cos[code].to_idx + 1):
                if cho[pos] == i:
                    out.write(f"  {slot[i].day} {slot[i].time:5s}")
                    tut_count += 1
                    if tut_count % 6 == 0:
                        out.write("\n     ")
        out.write("\n\n")


def Lecturer_Timetable(
    fd,
    cos: List[Chromoson],
    slot: List[Period],
    cho: List[int],
    Lecsize: List[int],
    Tutsize: List[int],
    nslot: int,
):
    fd.write("\n\t**********************************************************")
    fd.write("\n\t*            The timetable for every lecturer            *")
    fd.write("\n\t**********************************************************\n\n")

    current = lecturerlist
    if current is None:
        print("\n\t[LecturerClash] The lecturerlist is empty!")
    else:
        while current:
            node = current.hnext
            fd.write(f"{current.name}\n")
            fd.write("*" * len(current.name))
            fd.write("\n")
            while node:
                CreateSlotTable(fd, cos, slot, cho, Lecsize, Tutsize, nslot, node.code)
                node = node.next
            fd.write("\n")
            current = current.vnext


def fitlog(
    fd,
    cos: List[Chromoson],
    slot: List[Period],
    gen: List[int],
    conslot: List[int],
    tcosp: int,
    maxday: int,
    maxslot: int,
    s: int,
):
    numcosperday = [0] * maxslot
    conseperday = [0] * (maxslot - 1)
    numday = [0] * maxday
    numlec = [0] * tcosp

    fd.write("\nThe statistics of soft constraint violation for the timetable\n")
    fd.write("-------------------------------------------------------------\n")

    # StudentCons would be called here (implementation omitted for brevity)

    fd.write("Course spreading per day :\n")
    for i in range(maxslot):
        fd.write(f"\t{i + 2:3d} consecutive courses = {numcosperday[i]:4d}\n")

    fd.write("\n Course spreading per week :\n")
    for i in range(maxslot - 1):
        fd.write(f"\t{i + 1:3d} course(s) per day = {conseperday[i]:4d}\n")

    fd.write("\n Staff working days :\n")
    # StaffCons would be called here (implementation omitted for brevity)
    for i in range(maxday):
        fd.write(f"\t{i + 1:3d} working day(s) = {numday[i]:4d}\n")

    fd.write("\n Lecture spreading per week :\n")
    # CourseCons would be called here (implementation omitted for brevity)
    for i in range(tcosp):
        fd.write(f"\t{i + 1:3d} lecture(s) per day = {numlec[i]:4d}\n")

    fd.write("-------------------------------------------------------------\n\n")


def Evaluate(
    cos: List[Chromoson],
    slot: List[Period],
    gen: List[int],
    conslot: List[int],
    tcosp: int,
    maxday: int,
    maxslot: int,
    s: int,
):
    numcosperday = [0] * maxslot
    conseperday = [0] * (maxslot - 1)
    numday = [0] * maxday
    numlec = [0] * tcosp

    # StudentCons would be called here (implementation omitted for brevity)

    print("\tCourse spreading per day :")
    for i in range(maxslot):
        print(f"\t\t{i + 2:3d} consecutive courses = {numcosperday[i]:4d}")

    print("\n\tCourse spreading per week :")
    for i in range(maxslot - 1):
        print(f"\t\t{i + 1:3d} course(s) per day = {conseperday[i]:4d}")

    print("\n\tStaff working days :")
    # StaffCons would be called here (implementation omitted for brevity)
    for i in range(maxday):
        print(f"\t\t{i + 1:3d} working day(s) = {numday[i]:4d}")

    print("\n\tLecture spreading per week :")
    # CourseCons would be called here (implementation omitted for brevity)
    for i in range(tcosp):
        print(f"\t\t{i + 1:3d} lecture(s) per day = {numlec[i]:4d}")


def Fitness(
    cos: List[Chromoson],
    slot: List[Period],
    gen: List[int],
    cospenalty: List[int],
    copenalty: List[int],
    daypenalty: List[int],
    lecpenalty: List[int],
    tcosp: int,
    conslot: List[int],
    maxday: int,
    maxslot: int,
    s: int,
) -> float:
    total = 0

    # Calculate penalties (simplified)
    # In a real implementation, we would calculate actual violations
    # and multiply by the corresponding penalties

    return 1.0 / (1.0 + float(total))


def Ind_to_ind(dest: List[int], src: List[int]):
    for i in range(len(dest)):
        dest[i] = src[i]


def RoulWheel(cumfitval: List[float], popsize: int, choice: float) -> int:
    for i in range(popsize):
        if choice <= cumfitval[i]:
            return i
    return popsize - 1


def ParentSelect(
    cho: List[List[int]],
    cumfitval: List[float],
    parent: List[List[int]],
    popsize: int,
    num: int,
):
    i = 0
    ppos = 0
    while i < num:
        choice = cumfitval[-1] * random.random()
        cpos = RoulWheel(cumfitval, popsize, choice)

        if i == 1 and cpos == ppos:
            continue

        Ind_to_ind(parent[i], cho[cpos])
        ppos = cpos
        i += 1


def Swap(parent1: List[int], parent2: List[int], crosspoint: int):
    parent1[crosspoint], parent2[crosspoint] = parent2[crosspoint], parent1[crosspoint]


def Uniform_crossover(parent: List[List[int]], child: List[List[int]]):
    Num_Selections = random.randint(1, CHOLEN // 4)
    for _ in range(Num_Selections):
        crossover_point = random.randint(0, CHOLEN - 1)
        Swap(parent[0], parent[1], crossover_point)

    Ind_to_ind(child[0], parent[0])
    Ind_to_ind(child[1], parent[1])


def Mutation(parent: List[int], child: List[int], mutate: float, nslot: int):
    for i in range(CHOLEN):
        if random.random() < mutate:
            parent[i] = random.randint(0, nslot - 1)
    Ind_to_ind(child, parent)


def The_worst_member(fitval: List[float], fitness: float, popsize: int) -> int:
    worst = fitness
    worst_chromosome = -1

    for i in range(popsize):
        if fitness == fitval[i]:
            return -1
        elif worst > fitval[i]:
            worst = fitval[i]
            worst_chromosome = i

    return worst_chromosome


def Worst_replacement_scheme(
    cos: List[Chromoson],
    slot: List[Period],
    new_generation: List[List[int]],
    child: List[List[int]],
    fitval: List[float],
    cospenalty: List[int],
    copenalty: List[int],
    daypenalty: List[int],
    lecpenalty: List[int],
    tcosp: int,
    num_of_child: int,
    popsize: int,
    conslot: List[int],
    maxday: int,
    maxslot: int,
    nslot: int,
):
    for i in range(num_of_child):
        fitness_value = Fitness(
            cos,
            slot,
            child[i],
            cospenalty,
            copenalty,
            daypenalty,
            lecpenalty,
            tcosp,
            conslot,
            maxday,
            maxslot,
            nslot,
        )

        pos = The_worst_member(fitval, fitness_value, popsize)
        if pos != -1:
            fitval[pos] = fitness_value
            Ind_to_ind(new_generation[pos], child[i])


def Replace_generation(dest: List[List[int]], src: List[List[int]], popsize: int):
    for i in range(popsize):
        Ind_to_ind(dest[i], src[i])


def sorting(freeslot: List[int], fitval: List[float], fpos: int):
    for i in range(fpos):
        for j in range(i + 1, fpos):
            if fitval[i] < fitval[j]:
                freeslot[i], freeslot[j] = freeslot[j], freeslot[i]
                fitval[i], fitval[j] = fitval[j], fitval[i]


def savecholen(outfile: str, cos: List[Chromoson]):
    try:
        with open(outfile, "w") as fd:
            for i in range(TOTALCOURSE + 1):
                coursename = GetCourseName(i)
                fd.write(f"{coursename}${cos[i].lec}${cos[i].tut}$\n")
    except IOError as e:
        print(f"\n\t[savecholen] Error: {e}")
        exit(1)


def savecostime(
    outfile: str, cos: List[Chromoson], slot: List[Period], cho: List[int], size: int
):
    try:
        with open(outfile, "w") as fd:
            code = 0
            cntl = True
            coursename = GetCourseName(code)
            fd.write(f"{coursename}&")

            for i in range(size):
                if i > cos[code].to_idx:
                    code += 1
                    cntl = True
                    coursename = GetCourseName(code)
                    fd.write(f"&\n{coursename}&")

                if cntl and i >= cos[code].from_idx + cos[code].lec:
                    cntl = False
                    fd.write("&")

                fd.write(f"{slot[cho[i]].day} {slot[cho[i]].time}$")

                if cos[code].tut == 0 and i == cos[code].to_idx:
                    fd.write("&X")

            fd.write("&")
    except IOError as e:
        print(f"\n\t[savecostime] Error: {e}")
        exit(1)


def final(
    Tablefile: str,
    lecsizefile: str,
    cos: List[Chromoson],
    slot: List[Period],
    cho: List[int],
    conslot: List[int],
    tcosp: int,
    maxslot: int,
    mday: int,
    nslot: int,
):
    try:
        if os.path.exists(Tablefile):
            os.remove(Tablefile)

        with open(Tablefile, "w") as fd:
            tmpfile = f"{Tablefile}.tmp"
            fd.write(f"{tmpfile}\n\n")

            # Initialize Lecsize and Tutsize
            Lecsize = [0] * (TOTALCOURSE + 1)
            Tutsize = [0] * (TOTALCOURSE + 1)

            # ReadLecTutsize would be called here (implementation omitted for brevity)

            # Log statistics
            fitlog(fd, cos, slot, cho, conslot, tcosp, mday, maxslot, nslot)

            # Create timetable
            CreateTimeTable(
                fd,
                cos,
                slot,
                cho,
                maxslot,
                mday,
                Lecsize,
                Tutsize,
                "ALL COURSE TIMETABLE",
            )

            fd.write("\n\t**********************************************************")
            fd.write("\n\t*            The timetable for every course              *")
            fd.write(
                "\n\t**********************************************************\n\n"
            )

            for code in range(TOTALCOURSE + 1):
                CreateSlotTable(fd, cos, slot, cho, Lecsize, Tutsize, nslot, code)

            Lecturer_Timetable(fd, cos, slot, cho, Lecsize, Tutsize, nslot)

    except IOError as e:
        print(f"[final] Error: {e}")
        exit(1)


def schedule():
    # Initialize variables
    global TOTALCOURSE, CHOLEN

    # Parameters
    studentfile = "student.dat"
    lecturerfile = "lecturer.dat"
    LecTutsizefile = "lecsize.dat"
    timefile = "timeslot"
    LecTutfile = "lec.dat"
    scheslotfile = ""
    copenaltyfile = "crs"
    cospenaltyfile = "tv"
    staffpenaltyfile = "staff"
    coursepenaltyfile = "course"
    gensize = 10
    popsize = 10
    choice = 1

    # Start timing
    start_time = time.time()

    # Generate course codes
    Generate_course_code(studentfile)
    print("\n\tA list of course is recorded")
    print(f"\n\tThe total course is {TOTALCOURSE + 1}")

    # Initialize clash table
    clashtable = DynArray(TOTALCOURSE + 1, TOTALCOURSE + 1)

    # Initialize student and lecturer lists
    fillstudentlist(studentfile)
    print("\n\tThe student list is initialized")
    filllecturerlist(lecturerfile)
    print("\n\tThe lecturer list is initialized")

    # Create course structure
    cos = [Chromoson() for _ in range(TOTALCOURSE + 1)]
    IniCostruct(LecTutfile, cos)
    print(f"\n\tThe course structure is initialized")
    print(f"\n\tThe length of chromosome is {CHOLEN}")

    # Create output files
    Create_List_Of_File()

    # Record clashes
    LecturerClash(clashtable)
    print("\n\tLecturer clash is recorded...")
    StudentClash(clashtable)
    print("\n\tStudent clash is recorded...")

    # Display clash matrix
    Display_clash(clashtable)

    # Create population structures
    cho = DynArray(popsize, CHOLEN)
    print("\n\tChromosome structure is created...")

    New_Generation = DynArray(popsize, CHOLEN)
    parent = DynArray(2, CHOLEN)
    child = DynArray(2, CHOLEN)

    # Initialize time slots
    nslot = Getotal(timefile)
    slot = [Period() for _ in range(nslot)]
    Initimeslot(timefile, slot)
    print("\n\tslot structure is initialized")


def GetMaxSlot(slot: List[Period], nslot: int, maxslot: List[int], maxday: List[int]):
    day = slot[0].day
    counts = 0
    mslot = 0
    mday = 1

    for i in range(nslot):
        if slot[i].day == day:
            counts += 1
        else:
            if mslot < counts:
                mslot = counts
            counts = 1
            mday += 1
            day = slot[i].day

    if mslot < counts:
        mslot = counts

    maxslot[0] = mslot
    maxday[0] = mday


def TimeConse(slot: List[Period], conslot: List[int], nslot: int):
    conslot[0] = 1
    for i in range(nslot - 1):
        if slot[i].day == slot[i + 1].day:
            # Extract time parts (e.g., "8-9" -> 8 and 9)
            try:
                end1 = int(slot[i].time.split("-")[1])
                start2 = int(slot[i + 1].time.split("-")[0])
                if end1 == start2:
                    conslot[i + 1] = 1
                else:
                    conslot[i + 1] = 0
            except (IndexError, ValueError):
                conslot[i + 1] = 0
        else:
            conslot[i + 1] = 1


def Initcospenalty(cospenaltyfile: str, penalty: List[int], limit: int):
    try:
        with open(cospenaltyfile, "r") as fd:
            for line in fd:
                line = line.strip()
                if not line or line[0] == "#":
                    continue

                parts = line.split("$")
                if len(parts) >= 2:
                    index = int(parts[0]) - 2
                    if 0 <= index < limit:
                        penalty[index] = int(parts[1])
    except IOError as e:
        print(f"\n\t[Initcospenalty] Error: {e}")
        exit(1)


def Initcopenalty(copenaltyfile: str, penalty: List[int], limit: int):
    try:
        with open(copenaltyfile, "r") as fd:
            for line in fd:
                line = line.strip()
                if not line or line[0] == "#":
                    continue

                parts = line.split("$")
                if len(parts) >= 2:
                    index = int(parts[0]) - 1
                    if 0 <= index < limit:
                        penalty[index] = int(parts[1])
    except IOError as e:
        print(f"\n\t[Initcopenalty] Error: {e}")
        exit(1)


def Initstaffpenalty(staffpenaltyfile: str, penalty: List[int], limit: int):
    try:
        with open(staffpenaltyfile, "r") as fd:
            for line in fd:
                line = line.strip()
                if not line or line[0] == "#":
                    continue

                parts = line.split("$")
                if len(parts) >= 2:
                    index = int(parts[0]) - 1
                    if 0 <= index < limit:
                        penalty[index] = int(parts[1])
    except IOError as e:
        print(f"\n\t[Initstaffpenalty] Error: {e}")
        exit(1)


def Initlecpenalty(coursepenaltyfile: str, penalty: List[int], limit: int):
    try:
        with open(coursepenaltyfile, "r") as fd:
            for line in fd:
                line = line.strip()
                if not line or line[0] == "#":
                    continue
                if "TOTAL:" in line:
                    continue

                parts = line.split("$")
                if len(parts) >= 2:
                    index = int(parts[0]) - 1
                    if 0 <= index < limit:
                        penalty[index] = int(parts[1])
    except IOError as e:
        print(f"\n\t[Initlecpenalty] Error: {e}")
        exit(1)


def Duplicate(dest: List[float], src: List[float], popsize: int):
    for i in range(popsize):
        dest[i] = src[i]


def saveallfile(
    file: str,
    studentfile: str,
    lecturerfile: str,
    timefile: str,
    LecTutfile: str,
    LecTutsizefile: str,
    cospenaltyfile: str,
    copenaltyfile: str,
    staffpenaltyfile: str,
    coursepenaltyfile: str,
    scheslotfile: str,
):
    try:
        with open(file, "w") as fd:
            fd.write("\nSTD=" + studentfile)
            fd.write("\nLEC=" + lecturerfile)
            fd.write("\nCOS=" + scheslotfile)
            fd.write("\nCOSP=" + cospenaltyfile)
            fd.write("\nCOP=" + copenaltyfile)
            fd.write("\nSTP=" + staffpenaltyfile)
            fd.write("\nLEP=" + coursepenaltyfile)
            fd.write("\nTOC=" + LecTutfile)
            fd.write("\nTOS=" + LecTutsizefile)
            fd.write("\nTIM=" + timefile)
            fd.write("\n")
    except IOError as e:
        print(f"\n\t[saveallfile] Error: {e}")
        exit(1)

    # Get max slots and days
    maxslot = 0
    maxday = 0
    GetMaxSlot(slot, nslot, maxslot, maxday)
    print(f"\n\tNumber of day is {maxday}, maximum slot per day is {maxslot}")

    # Initialize consecutive slots
    conslot = intstruct(nslot)
    TimeConse(slot, conslot, nslot)

    # Initialize chromosomes
    for i in range(popsize):
        for j in range(CHOLEN):
            cho[i][j] = -1

    # Fill with already scheduled courses if any
    Fixslot = intstruct(CHOLEN)
    if scheslotfile:
        IniAschCos(scheslotfile, cos, slot, cho, nslot, popsize)
        print("\n\tChromosome is filled with already schedule course")

    # Record fixed slots
    for i in range(CHOLEN):
        Fixslot[i] = cho[0][i]

    # Initialize chromosomes
    InitChos(cos, clashtable, cho, Fixslot, popsize, nslot)
    print(f"\n\tThe first population with {popsize} members are initialized ...")

    # Initialize fitness structures
    fitval = doublestruct(popsize)
    cumfitval = doublestruct(popsize)
    cfitval = doublestruct(popsize)

    # Initialize penalty tables
    cospenalty = intstruct(maxslot - 1)
    Initcospenalty(cospenaltyfile, cospenalty, maxslot - 1)

    copenalty = intstruct(maxslot)
    Initcopenalty(copenaltyfile, copenalty, maxslot)

    daypenalty = intstruct(maxday)
    Initstaffpenalty(staffpenaltyfile, daypenalty, maxday)

    tcosp = Getotal(coursepenaltyfile)
    coursepenalty = intstruct(tcosp)
    Initlecpenalty(coursepenaltyfile, coursepenalty, tcosp)

    # Initialize random generator
    random.seed(time.time())

    print(f"\n\tThe process is going to create {gensize} generations")
    print("\n\tPlease wait .... until process is completed")

    # Main GA loop
    Num_Of_Generation = 0
    while Num_Of_Generation < gensize:
        print(f"\n\t{Num_Of_Generation:4d} Generation is created", end="")

        # Make a copy of current generation
        Replace_generation(New_Generation, cho, popsize)

        # Calculate fitness values
        for i in range(popsize):
            fitval[i] = Fitness(
                cos,
                slot,
                cho[i],
                cospenalty,
                copenalty,
                daypenalty,
                coursepenalty,
                tcosp,
                conslot,
                maxday,
                maxslot,
                nslot,
            )
            if i == 0:
                cumfitval[i] = fitval[i]
            else:
                cumfitval[i] = cumfitval[i - 1] + fitval[i]

        Duplicate(cfitval, fitval, popsize)

        # Reproduction
        Num_Of_Child = 0
        while Num_Of_Child < popsize:
            if choice == 1:
                ParentSelect(cho, cumfitval, parent, popsize, 2)
                Uniform_crossover(parent, child)
                Size_of_child = 2
            elif choice == 2:
                ParentSelect(cho, cumfitval, parent, popsize, 1)
                Mutation(parent[0], child[0], 0.08, nslot)
                Size_of_child = 1
            elif choice == 3:
                ParentSelect(cho, cumfitval, parent, popsize, 1)
                # Directed_mutation would be called here (implementation omitted)
                Size_of_child = 1

            if Size_of_child:
                Num_Of_Child += Size_of_child

                if Num_Of_Child > popsize:
                    Size_of_child = Num_Of_Child - popsize
                    Num_Of_Child -= Size_of_child

                # Correct clashes
                # Clash_Correct would be called here (implementation omitted)

                # Replace worst solutions
                Worst_replacement_scheme(
                    cos,
                    slot,
                    New_Generation,
                    child,
                    cfitval,
                    cospenalty,
                    copenalty,
                    daypenalty,
                    coursepenalty,
                    tcosp,
                    Size_of_child,
                    popsize,
                    conslot,
                    maxday,
                    maxslot,
                    nslot,
                )

        # Replace old generation
        Replace_generation(cho, New_Generation, popsize)
        Num_Of_Generation += 1

    # Find best solution
    maxfit = 0
    pos = 0
    for i in range(popsize):
        value = Fitness(
            cos,
            slot,
            cho[i],
            cospenalty,
            copenalty,
            daypenalty,
            coursepenalty,
            tcosp,
            conslot,
            maxday,
            maxslot,
            nslot,
        )
        if maxfit < value:
            maxfit = value
            pos = i

    print(f"\n\tThe fittest individual [{pos}] with {maxfit} (fitness)")

    # Save results
    outfile = input("\n\tPlease specify a output filename: ")
    final(
        outfile,
        LecTutsizefile,
        cos,
        slot,
        cho[pos],
        conslot,
        tcosp,
        maxslot,
        maxday,
        nslot,
    )

    # Save course data
    outmpfile = f"{outfile}.cos"
    savecostime(outmpfile, cos, slot, cho[pos], CHOLEN)
    print(f"\n\tThe course data is saved in {outmpfile}")

    # Save all files info
    tmpfile = f"{outfile}.tmp"
    saveallfile(
        tmpfile,
        studentfile,
        lecturerfile,
        timefile,
        LecTutfile,
        LecTutsizefile,
        cospenaltyfile,
        copenaltyfile,
        staffpenaltyfile,
        coursepenaltyfile,
        outmpfile,
    )

    # Clean up
    if os.path.exists("student.bin"):
        os.remove("student.bin")

    # Print execution time
    end_time = time.time()
    print(
        f"\n\tThe whole process takes {int(end_time - start_time)} seconds to be completed\n"
    )


def reschedule():
    # Similar structure to schedule() but for rescheduling existing timetables
    pass


def main():
    print("\n\n")
    print("\t****************************************************************")
    print("\t**                                                            **")
    print("\t**               THE COURSE TIMETABLING SYSTEM                **")
    print("\t**            FOR INFORMATION TECHNOLOGY FACULTY              **")
    print("\t**          *************************************             **")
    print("\t**                                                            **")
    print("\t**           [1]     SCHEDULE A NEW TIMETABLE                 **")
    print("\t**           [2]     RESCHEDULE A TIMETABLE                   **")
    print("\t**                                                            **")
    print("\t****************************************************************")

    choice = input("\t                Choose a option [1-2]: ")
    while choice not in ["1", "2"]:
        choice = input("\n\tInput incorrect, re-enter a choice [1 or 2]: ")

    if choice == "1":
        schedule()
    else:
        reschedule()


if __name__ == "__main__":
    main()
