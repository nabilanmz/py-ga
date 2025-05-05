import os
import sys
import re
import random
import time
from typing import List, Dict, Tuple, Optional, Union
import copy

# Constants from ga.h
MAXLECTURERNAME = 20
MAXCOURSENAME = 16
MAXDAYCHAR = 4
MAXSLOTCHAR = 8
BUFSIZE = 100000

# Global variables
TOTALCOURSE = 0
CHOLEN = 0

class Code:
    def __init__(self):
        self.code = 0
        self.next = None

class Lecturer:
    def __init__(self):
        self.name = ""
        self.hnext = None  # Pointer to first course code
        self.vnext = None  # Pointer to next lecturer

class Course:
    def __init__(self):
        self.name = ""
        self.code = 0
        self.count = 0
        self.next = None

class Chromoson:
    def __init__(self):
        self.frm = 0  # 'from' is a keyword in Python
        self.to = 0
        self.tut = 0
        self.lec = 0

class Period:
    def __init__(self):
        self.day = ""
        self.time = ""

# Global linked list headers
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

def DynArray(rows: int, cols: int) -> List[List[int]]:
    try:
        # Create a 2D array initialized to 0
        return [[0 for _ in range(cols)] for _ in range(rows)]
    except MemoryError:
        print("No heap space for array")
        exit(1)

def DynFree(arr: List[List[int]]):
    # In Python, garbage collection is automatic, but we can clear references
    del arr

def Match_Lecturer(lecturername: str) -> bool:
    current = lecturerlist
    while current:
        if current.name.upper() == lecturername.upper():
            return True
        current = current.vnext
    return False

def AddLecturernode(lecturername: str, code: int) -> bool:
    global lecturerlist

    newcoursecode = Code()
    newcoursecode.code = code
    newcoursecode.next = None

    current = lecturerlist
    while current:
        if current.name.upper() == lecturername.upper():
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
    newlecturer.name = lecturername.upper()
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
    global TOTALCOURSE

    try:
        with open(lecturerfile, "r") as fd1, open("CourseLecturer.o", "w") as fd2:
            for line in fd1:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Split record into course and lecturers
                parts = line.split('&')
                coursename = parts[0].strip().upper()
                if len(parts) < 2:
                    continue

                lecturer_part = parts[1]
                if not lecturer_part:
                    continue

                # Get course code
                code = Get_Course_Code(coursename)
                if code == -1:
                    print(f"\n\t[filllecturerlist] Invalid course code: {coursename}!")
                    exit(1)

                fd2.write(f"{code}\t{coursename}\t{lecturer_part}\n")

                # Split lecturers by $
                lecturers = lecturer_part.split('$')
                for lecturername in lecturers:
                    if not lecturername:
                        continue

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
            for line in fd1:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Split student record
                parts = line.split('$')
                studentname = parts[0]

                # Process each course
                for coursename in parts[1:]:
                    if not coursename:
                        continue

                    coursename = coursename.upper()
                    code = Get_Course_Code(coursename)
                    if code == -1:
                        print(f"\n\t[fillstudentlist] Invalid course code: {coursename}!")
                        exit(1)

                    buf += f"{code} "
                buf += "$"

            # Write the buffer to binary file
            fd2.write(buf.encode('utf-8'))
            fd2.write("&".encode('utf-8'))
    except IOError as e:
        print(f"\n\t[fillstudentlist] Error: {e}")
        exit(1)

def Match_Course(coursename: str) -> bool:
    global head

    current = head
    while current:
        if current.name.upper() == coursename.upper():
            current.count += 1
            return False
        current = current.next
    return True

def AddCourse(coursename: str):
    global head, TOTALCOURSE

    newcourse = Course()
    newcourse.name = coursename.upper()
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
        if current.name.upper() == coursename.upper():
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
            for line in infile:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Split by $ delimiter
                parts = line.split('$')
                for coursename in parts:
                    if not coursename:
                        continue

                    coursename = coursename.upper()
                    if Match_Course(coursename):
                        AddCourse(coursename)
    except IOError as e:
        print(f"\n\t[Generate_course_code] Error: {e}")
        exit(1)

def StudentClash(clashtable: List[List[int]]):
    try:
        with open("student.bin", "rb") as fd:
            buf = fd.read().decode('utf-8')

            z = 0
            while z < len(buf) and buf[z] != '&':
                course = []
                j = 0
                codestr = ""

                while buf[z] != '$' and z < len(buf) and buf[z] != '&':
                    if buf[z] != ' ':
                        codestr += buf[z]
                    else:
                        if codestr:
                            course.append(int(codestr))
                            codestr = ""
                    z += 1

                if codestr:  # Add the last course if exists
                    course.append(int(codestr))

                # Record clashes between all pairs of courses for this student
                for k in range(len(course)):
                    for l in range(k + 1, len(course)):
                        clashtable[course[l]][course[k]] = 1
                        clashtable[course[k]][course[l]] = 1

                if buf[z] == '$':
                    z += 1
    except IOError as e:
        print(f"[StudentClash] Error: {e}")
        exit(1)

def CourseCredit(coursecode: int, coursename: str) -> str:
    # Get the last character as credit value
    val = int(coursename[-1])

    if coursename[2].upper() == 'P':  # Project course
        temp = val / 2.0
        return f"{coursecode}\t{coursename}\t\t2 semesters({temp:.1f})"
    else:
        return f"{coursecode}\t{coursename}\t\t1 semester({val})"

def IniCostruct(LecTutfile: str, cos: List[Chromoson]):
    global CHOLEN

    try:
        with open(LecTutfile, "r") as fd:
            for line in fd:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split('$')
                coursename = parts[0].strip().upper()
                lecsize = parts[1]
                tutsize = parts[2]

                coursecode = Get_Course_Code(coursename)
                if coursecode == -1:
                    print(f"\n\t[IniCostruct] Invalid course code: {coursename}!")
                    exit(1)

                # Initialize course information
                cos[coursecode].lec = int(lecsize)
                cos[coursecode].tut = int(tutsize)

        # Initialize course structure ranges
        for coursecode in range(TOTALCOURSE + 1):
            if coursecode == 0:
                cos[coursecode].frm = 0
            else:
                cos[coursecode].frm = cos[coursecode - 1].to + 1
            cos[coursecode].to = cos[coursecode].frm + cos[coursecode].lec + cos[coursecode].tut - 1

        # Length of the chromoson structure
        CHOLEN = cos[TOTALCOURSE].to + 1
    except IOError as e:
        print(f"\n\t[IniCostruct] Error: {e}")
        exit(1)

def Create_List_Of_File():
    global head

    try:
        with open("ListOfCourse.o", "w") as fd1, \
             open("NumOfStudent.o", "w") as fd2, \
             open("CourseCredit.o", "w") as fd3:

            current = head
            if current is None:
                print("\n\t[Create_List_Of_File] The course list is empty!")
                return

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
    global lecturerlist

    current = lecturerlist
    if current is None:
        print("\n\t[LecturerClash] The lecturerlist is empty!")
        return

    while current:
        course = []
        node = current.hnext
        while node:
            course.append(node.code)
            node = node.next

        # Record clashes between all pairs of courses for this lecturer
        for k in range(len(course)):
            for l in range(k + 1, len(course)):
                clashtable[course[l]][course[k]] = 1
                clashtable[course[k]][course[l]] = 1

        current = current.vnext

def Getotal(anyfile: str) -> int:
    try:
        with open(anyfile, "r") as fd:
            for line in fd:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                if "TOTAL:" in line:
                    parts = line.split(':')
                    return int(parts[1])
        return -1
    except IOError as e:
        print(f"\n\t[Getotal] Error: {e}")
        exit(1)

def Initimeslot(timefile: str, slot: List[Period]) -> int:
    nslot = 0
    try:
        with open(timefile, "r") as fd:
            for line in fd:
                line = line.strip()
                if not line or line.startswith('#') or "TOTAL:" in line:
                    continue

                parts = line.split('\t')
                day = parts[0].strip().upper()

                for time_slot in parts[1:]:
                    if not time_slot:
                        continue

                    slot[nslot].day = day
                    slot[nslot].time = time_slot.strip()
                    nslot += 1
        return nslot
    except IOError as e:
        print(f"\n\t[Initimeslot] Error: {e}")
        exit(1)

def Matchtimeslot(timestr: str, slot: List[Period], nslot: int) -> int:
    for i in range(nslot):
        slot_str = f"{slot[i].day} {slot[i].time}"
        if slot_str.startswith(timestr):
            return i
    return -1

def partialconvert(timeptr: str) -> str:
    parts = timeptr.split()
    if len(parts) != 2:
        return timeptr.upper()

    day = parts[0].upper()
    time_part = parts[1]
    return f"{day} {time_part}"

def Initchorec(cos: List[Chromoson], slot: List[Period], coursecode: int,
               rec: str, cho: List[List[int]], nslot: int, popsize: int, type_: int):
    if type_:
        start = cos[coursecode].frm + cos[coursecode].lec
        end = start + cos[coursecode].tut - 1
    else:
        start = cos[coursecode].frm
        end = start + cos[coursecode].lec - 1

    time_slots = rec.split('$')
    for timeptr in time_slots:
        if not timeptr:
            continue

        if start > end:
            print("\n\t[Initchorec] Invalid range!")
            exit(1)

        timestr = partialconvert(timeptr)
        timecode = Matchtimeslot(timestr, slot, nslot)
        if timecode == -1:
            print("\n\t[Initchorec] Invalid time slot!")
            exit(1)

        # Initialize the genes with already scheduled time slot
        for row in range(popsize):
            cho[row][start] = timecode
        start += 1

def IniAschCos(scheslotfile: str, cos: List[Chromoson], slot: List[Period],
               cho: List[List[int]], nslot: int, popsize: int):
    try:
        with open(scheslotfile, "r") as fd:
            for line in fd:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split('&')
                coursename = parts[0].strip().upper()
                lec = parts[1] if len(parts) > 1 else "X"
                tut = parts[2] if len(parts) > 2 else "X"

                coursecode = Get_Course_Code(coursename)
                if coursecode == -1:
                    print(f"\n\t[IniAschCos] Invalid course code: {coursename}!")
                    exit(1)

                if lec != "X":
                    Initchorec(cos, slot, coursecode, lec, cho, nslot, popsize, 0)
                if tut != "X":
                    Initchorec(cos, slot, coursecode, tut, cho, nslot, popsize, 1)
    except IOError as e:
        print(f"\n\t[IniAschCos] Error: {e}")
        exit(1)

def InitChos(cos: List[Chromoson], clashtable: List[List[int]],
             cho: List[List[int]], Fixslot: List[int], popsize: int, nslot: int):
    for row in range(popsize):
        col = 0
        code = 0
        first_flag = True
        tut_flag = True

        freeslot = intstruct(nslot)
        clashslot = intstruct(nslot)

        while col < CHOLEN:
            cntl = False

            # Determine course code by position in chromosome
            if col == -1:
                print("\n[InitChos] Invalid gene position")
                exit(1)

            if col > cos[code].to:
                for tpos in range(nslot):
                    clashslot[tpos] = 0
                    freeslot[tpos] = 0
                code += 1
                tut_flag = True
                cntl = True
            elif col < cos[code].frm:
                for tpos in range(nslot):
                    clashslot[tpos] = 0
                    freeslot[tpos] = 0
                code -= 1
                tut_flag = True
                cntl = True

            if first_flag:
                cntl = True
                first_flag = False
                tut_flag = True

            if tut_flag and col > (cos[code].frm + cos[code].lec - 1):
                for tpos in range(nslot):
                    clashslot[tpos] = 0
                    freeslot[tpos] = 0
                cntl = True
                tut_flag = False

            if Fixslot[col] == -1:  # Gene is available for new time slot
                if cntl:
                    # Check clash with other courses
                    for cpos in range(TOTALCOURSE + 1):
                        if code == cpos:
                            continue

                        if clashtable[cpos][code]:
                            # Check clash between:
                            # 1. lecture with lecture
                            # 2. lecture with tutorial
                            # Avoid tutorial with tutorial
                            for gpos in range(cos[cpos].frm, cos[cpos].to + 1):
                                if cho[row][gpos] == -1:
                                    continue
                                if (col > (cos[code].frm + cos[code].lec - 1) and \
                                   (gpos > (cos[cpos].frm + cos[cpos].lec - 1)):
                                    break
                                clashslot[cho[row][gpos]] = 1

                # Filter out time slots assigned for course
                for gpos in range(cos[code].frm, cos[code].to + 1):
                    if cho[row][gpos] == -1:
                        continue
                    clashslot[cho[row][gpos]] = 1

                # Find free time slots
                fpos = 0
                for tpos in range(nslot):
                    if clashslot[tpos] == 0:
                        freeslot[fpos] = tpos
                        fpos += 1

                if fpos > 0:
                    # Select a time slot randomly from free slots
                    tpos = random.randint(0, fpos - 1)
                    cho[row][col] = freeslot[tpos]
                    col += 1
                else:
                    if Backtracking(clashtable, cos, cho[row], Fixslot, code, col, nslot):
                        col += 1
                    else:
                        col -= 1
            else:
                col += 1

        del freeslot
        del clashslot

def Backtracking(clashtable: List[List[int]], cos: List[Chromoson],
                 cho: List[int], Fixslot: List[int], code: int, col: int, nslot: int) -> bool:
    timeslot = intstruct(nslot)
    tmpslot = intstruct(nslot)
    freeslot = intstruct(nslot)
    tmpcho = intstruct(CHOLEN)

    # Copy chromosome value into temporary buffer
    Ind_to_ind(tmpcho, cho)

    # Determine number of clashes associated with the time slot
    for cpos in range(TOTALCOURSE + 1):
        if code == cpos:
            continue

        if clashtable[cpos][code]:
            for gpos in range(cos[cpos].frm, cos[cpos].to + 1):
                if tmpcho[gpos] == -1:
                    continue
                if (col > (cos[code].frm + cos[code].lec - 1)) and \
                   (gpos > (cos[cpos].frm + cos[cpos].lec - 1)):
                    continue
                else:
                    timeslot[cho[gpos]] += 1

    for gpos in range(cos[code].frm, cos[code].to + 1):
        if tmpcho[gpos] == -1:
            continue
        timeslot[tmpcho[gpos]] += 1

    NumofClash = 1
    NumofSlot = 0
    while NumofSlot < nslot:
        tpos = 0
        for pos in range(nslot):
            if NumofClash == timeslot[pos]:
                tmpslot[tpos] = pos
                tpos += 1

        while tpos > 0:
            randslot = random.randint(0, tpos - 1)
            npos = 0
            cntl = True

            for gpos in range(CHOLEN):
                if tmpcho[gpos] == tmpslot[randslot]:
                    gfreeslot(clashtable, cos, tmpcho, Fixslot, gpos, nslot, freeslot, npos)

                    if npos > 0:
                        randslot = random.randint(0, npos - 1)
                        tmpcho[col] = tmpcho[gpos]
                        tmpcho[gpos] = freeslot[randslot]
                    else:
                        cntl = False
                        break

                    npos += 1

                if not cntl or npos >= NumofClash:
                    break

            if cntl:
                Ind_to_ind(cho, tmpcho)
                del timeslot, tmpslot, freeslot, tmpcho
                return True

            # Remove the used slot
            for i in range(randslot, tpos - 1):
                tmpslot[i] = tmpslot[i + 1]

            tpos -= 1
            NumofSlot += NumofClash

        NumofClash += 1

    del timeslot, tmpslot, freeslot, tmpcho
    return False

def gfreeslot(clashtable: List[List[int]], cos: List[Chromoson],
              cho: List[int], Fixslot: List[int], col: int, nslot: int,
              freeslot: List[int], fpos: int):
    clashslot = intstruct(nslot)

    # Determine course code
    code = 0
    for c in range(TOTALCOURSE + 1):
        if (col >= cos[c].frm) and (col <= cos[c].to):
            code = c
            break

    # Check clash between courses
    for cpos in range(TOTALCOURSE + 1):
        if code == cpos:
            continue

        if clashtable[cpos][code]:
            for gpos in range(cos[cpos].frm, cos[cpos].to + 1):
                if cho[gpos] == -1:
                    continue
                if (col > (cos[code].frm + cos[code].lec - 1)) and \
                   (gpos > (cos[cpos].frm + cos[cpos].lec - 1)):
                    break
                if Fixslot[gpos] == -1:
                    clashslot[cho[gpos]] = 1

    # Check clash between course items
    for gpos in range(cos[code].frm, cos[code].to + 1):
        if cho[gpos] == -1:
            continue
        if Fixslot[gpos] == -1:
            clashslot[cho[gpos]] = 1

    # Get free slots for particular course item
    fpos[0] = 0
    for tpos in range(nslot):
        if clashslot[tpos] == 0:
            freeslot[fpos[0]] = tpos
            fpos[0] += 1

    del clashslot

def ReadLecTutsize(Lecsizefile: str, Lec: List[int], Tut: List[int]):
    try:
        with open(Lecsizefile, "r") as fd:
            for line in fd:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split('$')
                coursename = parts[0].strip().upper()
                lecsize = parts[1]
                tutsize = parts[2]

                coursecode = Get_Course_Code(coursename)
                if coursecode == -1:
                    print(f"\n\t[ReadLecTutsize] Invalid course code: {coursename}!")
                    exit(1)

                # Initialize course information
                Lec[coursecode] = int(lecsize)
                Tut[coursecode] = int(tutsize)
    except IOError as e:
        print(f"\n\t[ReadLecTutsize] Error: {e}")
        exit(1)

def CreateTimeTable(out, cos: List[Chromoson], slot: List[Period],
                   cho: List[int], nslot: int, mday: int, lecsize: List[int],
                   tutsize: List[int], title: str):
    slotable = intstruct(CHOLEN)

    # Initialize temporary slot timetable array
    for pos in range(CHOLEN):
        slotable[pos] = 0

    out.write(f"{title}\n")

    # Line drawing
    out.write("-------------")
    for pos in range(mday):
        out.write("---------------")
    out.write("\n")
    out.write("|\t")

    # Print day of week
    days = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
    for pos in range(mday):
        out.write(f"|      {days[pos]}      ")
    out.write("|\n")

    out.write("-------------")
    for pos in range(mday):
        out.write("---------------")
    out.write("\n")

    times = ["8-9", "9-10", "10-11", "11-12", "12-1", "1-2", "2-3", "3-4", "4-5"]

    for i in range(nslot):
        counts = 0
        cntlf = 0

        out.write(f"|{times[i]}")

        # Determine corresponding time slot in chromosome
        for col in range(CHOLEN):
            if slot[cho[col]].time.startswith(times[i]):
                slotable[col] = 1
                counts += 1

        if counts == 0:
            for day in range(mday):
                if day == 0:
                    if cntlf == 0:
                        out.write("\t")
                        cntlf = 1
                    else:
                        out.write("\n|\t")
                out.write("|               ")
            out.write("|")
        else:
            while counts > 0:
                for day in range(mday):
                    code = 0
                    if day == 0:
                        if cntlf == 0:
                            out.write("\t")
                            cntlf = 1
                        else:
                            out.write("\n|\t")

                    cntl = False
                    for col in range(CHOLEN):
                        gpos = cho[col]
                        if col > cos[code].to:
                            code += 1

                        if slotable[col]:
                            if slot[gpos].day.startswith(days[day]):
                                cntl = True
                                slotable[col] = 0
                                counts -= 1
                                break

                    if cntl:
                        # Display the course name
                        coursename = GetCourseName(code)
                        if col <= (cos[code].frm + cos[code].lec - 1):
                            out.write(f"|{coursename}(L)[{lecsize[code]:3d}]")
                        else:
                            out.write(f"|{coursename}(T)[{tutsize[code]:3d}]")
                    else:
                        out.write("|               ")
                out.write("|")
        out.write("\n")
        out.write("-------------")
        for pos in range(mday):
            out.write("---------------")
        out.write("\n")

    out.write("\n\n")
    del slotable

def CreateSlotTable(out, cos: List[Chromoson], slot: List[Period],
                   cho: List[int], Lecsize: List[int], Tutsize: List[int],
                   nslot: int, code: int):
    coursename = GetCourseName(code)
    out.write(f"{coursename}")

    if cos[code].lec > 0:
        i = 0
        gpos = 0
        out.write(f"\nLec ({Lecsize[code]:3d})")
        while i < nslot:
            if gpos == cos[code].lec:
                break

            for pos in range(cos[code].frm, cos[code].frm + cos[code].lec):
                if cho[pos] == i:
                    out.write(f"  {slot[i].day} {slot[i].time:5s}")
                    gpos += 1
                    if (gpos % 4) == 0:
                        out.write("\n    ")
            i += 1

    if cos[code].tut > 0:
        i = 0
        gpos = 0
        out.write(f"\nTut ({Tutsize[code]:3d})")
        while i < nslot:
            if gpos == cos[code].tut:
                break

            for pos in range(cos[code].frm + cos[code].lec, cos[code].to + 1):
                if cho[pos] == i:
                    out.write(f"  {slot[i].day} {slot[i].time:5s}")
                    gpos += 1
                    if (gpos % 6) == 0:
                        out.write("\n     ")
            i += 1

    out.write("\n\n")

def Lecturer_Timetable(fd, cos: List[Chromoson], slot: List[Period],
                       cho: List[int], Lecsize: List[int], Tutsize: List[int], nslot: int):
    fd.write("\n\t**********************************************************")
    fd.write("\n\t*            The timetable for every lecturer            *")
    fd.write("\n\t**********************************************************\n\n")

    current = lecturerlist
    if current is None:
        print("\n\t[Lecturer_Timetable] The lecturerlist is empty!")
        return

    while current:
        node = current.hnext
        fd.write(f"{current.name}\n")
        fd.write("*" * len(current.name) + "\n")

        while node:
            CreateSlotTable(fd, cos, slot, cho, Lecsize, Tutsize, nslot, node.code)
            node = node.next

        fd.write("\n")
        current = current.vnext

def fitlog(fd, cos: List[Chromoson], slot: List[Period], gen: List[int],
           conslot: List[int], tcosp: int, maxday: int, maxslot: int, s: int):
    numcosperday = intstruct(maxslot)
    conseperday = intstruct(maxslot - 1)
    numday = intstruct(maxday)
    numlec = intstruct(tcosp)

    fd.write("\nThe statistics of soft constraint violation for the timetable\n")
    fd.write("-------------------------------------------------------------\n")
    StudentCons(cos, slot, gen, conseperday, numcosperday, conslot, maxslot, s)

    fd.write("Course spreading per day :\n")
    for i in range(maxslot):
        fd.write(f"\t{i + 2:3d} consecutive courses = {numcosperday[i]:4d}\n")

    fd.write("\n Course spreading per week :\n")
    for i in range(maxslot - 1):
        fd.write(f"\t{i + 1:3d} course(s) per day = {conseperday[i]:4d}\n")

    fd.write("\n Staff working days :\n")
    StaffCons(cos, slot, gen, numday, maxslot, s)
    for i in range(maxday):
        fd.write(f"\t{i + 1:3d} working day(s) = {numday[i]:4d}\n")

    fd.write("\n Lecture spreading per week :\n")
    CourseCons(cos, slot, gen, numlec, s)
    for i in range(tcosp):
        fd.write(f"\t{i + 1:3d} lecture(s) per day = {numlec[i]:4d}\n")

    fd.write("-------------------------------------------------------------\n\n")

    del numcosperday, conseperday, numday, numlec

def Evaluate(cos: List[Chromoson], slot: List[Period], gen: List[int],
             conslot: List[int], tcosp: int, maxday: int, maxslot: int, s: int):
    numcosperday = intstruct(maxslot)
    conseperday = intstruct(maxslot - 1)
    numday = intstruct(maxday)
    numlec = intstruct(tcosp)

    StudentCons(cos, slot, gen, conseperday, numcosperday, conslot, maxslot, s)
    print("\tCourse spreading per day :")
    for i in range(maxslot):
        print(f"\t\t{i + 2:3d} consecutive courses = {numcosperday[i]:4d}")

    print("\n\tCourse spreading per week :")
    for i in range(maxslot - 1):
        print(f"\t\t{i + 1:3d} course(s) per day = {conseperday[i]:4d}")

    print("\n\tStaff working days :")
    StaffCons(cos, slot, gen, numday, maxslot, s)
    for i in range(maxday):
        print(f"\t\t{i + 1:3d} working day(s) = {numday[i]:4d}")

    print("\n\tLecture spreading per week :")
    CourseCons(cos, slot, gen, numlec, s)
    for i in range(tcosp):
        print(f"\t\t{i + 1:3d} lecture(s) per day = {numlec[i]:4d}")

    del numcosperday, conseperday, numday, numlec

def Ind_to_ind(dest: List[int], src: List[int]):
    for i in range(len(dest)):
        dest[i] = src[i]

def RoulWheel(cumfitval: List[float], popsize: int, choice: float) -> int:
    for i in range(popsize):
        if choice <= cumfitval[i]:
            return i
    return popsize - 1  # Fallback

def ParentSelect(cho: List[List[int]], cumfitval: List[float],
                parent: List[List[int]], popsize: int, num: int):
    i = 0
    ppos = 0

    while i < num:
        choice = random.uniform(0, cumfitval[popsize - 1])

        # Select parent
        cpos = RoulWheel(cumfitval, popsize, choice)

        # Avoid choosing the same parent
        if i == 1 and cpos == ppos:
            continue

        Ind_to_ind(parent[i], cho[cpos])
        ppos = cpos
        i += 1

def Swap(parent1: List[int], parent2: List[int], crosspoint: int):
    temp = parent1[crosspoint]
    parent1[crosspoint] = parent2[crosspoint]
    parent2[crosspoint] = temp

def Display(cho: List[int]):
    for i in range(len(cho)):
        print(f"{cho[i]} ", end='')
    print()

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
            val = random.randint(0, nslot - 1)
            parent[i] = val
    Ind_to_ind(child, parent)

def The_worst_member(fitval: List[float], fitness: float, popsize: int) -> int:
    worst_chromosome = -1
    worst = fitness

    for i in range(popsize):
        # Ignore duplicated or similar individual selection
        if fitness == fitval[i]:
            return -1
        elif worst > fitval[i]:
            worst = fitval[i]
            worst_chromosome = i

    return worst_chromosome

def Worst_replacement_scheme(cos: List[Chromoson], slot: List[Period],
                            new_generation: List[List[int]], child: List[List[int]],
                            fitval: List[float], cospenalty: List[int], copenalty: List[int],
                            daypenalty: List[int], lecpenalty: List[int], tcosp: int,
                            num_of_child: int, popsize: int, conslot: List[int],
                            maxday: int, maxslot: int, nslot: int):
    for i in range(num_of_child):
        fitness_value = Fitness(cos, slot, child[i], cospenalty, copenalty, daypenalty,
                               lecpenalty, tcosp, conslot, maxday, maxslot, nslot)

        # Retrieve the worst individual and ignore duplicated solutions
        pos = The_worst_member(fitval, fitness_value, popsize)
        if pos != -1:
            # Replace fitness table with new individual's fitness value
            fitval[pos] = fitness_value
            Ind_to_ind(new_generation[pos], child[i])

def Replace_generation(dest: List[List[int]], src: List[List[int]], popsize: int):
    for i in range(popsize):
        Ind_to_ind(dest[i], src[i])

def show(cho: List[List[int]], popsize: int):
    for i in range(popsize):
        print(f"Member: {i}")
        for j in range(CHOLEN):
            print(f"{cho[i][j]} ", end='')
        print("\n")

def Duplicate(dest: List[float], src: List[float], popsize: int):
    for i in range(popsize):
        dest[i] = src[i]

def compare(arr: List[int], result: List[int], arrsize: int, resize: int, order: int):
    if order:  # Find maximum values
        for i in range(resize):
            max_val = -1
            pos = 0
            for j in range(arrsize):
                if arr[j] == -1:
                    continue
                if max_val < arr[j]:
                    max_val = arr[j]
                    pos = j
            result[i] = pos
            arr[pos] = -1
    else:  # Find minimum values
        for i in range(resize):
            min_val = float('inf')
            pos = 0
            for j in range(arrsize):
                if arr[j] == -1:
                    continue
                if min_val > arr[j]:
                    min_val = arr[j]
                    pos = j
            result[i] = pos
            arr[pos] = -1

def Violation(cos: List[Chromoson], slot: List[Period], gen: List[int],
              chopenalty: List[int], slotpenalty: List[int], copenalty: List[int],
              cospenalty: List[int], conslot: List[int], maxslot: int, nslot: int):
    try:
        with open("student.bin", "rb") as fd:
            buf = fd.read().decode('utf-8')

            z = 0
            while z < len(buf) and buf[z] != '&':
                course = []
                j = 0
                codestr = ""

                while buf[z] != '$' and z < len(buf) and buf[z] != '&':
                    if buf[z] != ' ':
                        codestr += buf[z]
                    else:
                        if codestr:
                            course.append(int(codestr))
                            codestr = ""
                    z += 1

                if codestr:  # Add the last course if exists
                    course.append(int(codestr))

                # Process the student's courses
                if course:
                    # Initialize tracking variables
                    weekday = slot[0].day
                    consecutive = 0
                    slotnum = 0
                    cosnum = 0
                    cosqueue = []
                    coqueue = []
                    chopos = []

                    # Record positions of all courses for this student
                    for coursecode in course:
                        for pos in range(cos[coursecode].frm, cos[coursecode].to + 1):
                            chopos.append(pos)

                    # Process time slots
                    j = 0
                    while j < nslot:
                        # Check if still in same day
                        if slot[j].day == weekday:
                            if gen[j] in chopos:
                                # Check if consecutive with previous slot
                                if conslot[j] == 0 and j > 0 and gen[j-1] in chopos:
                                    # Assign consecutive penalty
                                    if consecutive > 1:
                                        for pos in range(slotnum):
                                            for gpos in chopos:
                                                if cosqueue[pos] == gen[gpos]:
                                                    chopenalty[gpos] += cospenalty[consecutive - 2]
                                            slotpenalty[cosqueue[pos]] += cospenalty[consecutive - 2]
                                    slotnum = 0
                                    consecutive = 0

                                # Track this slot
                                if slotnum < maxslot and cosnum < maxslot:
                                    cosqueue.append(j)
                                    coqueue.append(j)
                                    slotnum += 1
                                    cosnum += 1
                                    consecutive += 1
                            else:
                                if consecutive > 1:
                                    for pos in range(slotnum):
                                        for gpos in chopos:
                                            if cosqueue[pos] == gen[gpos]:
                                                chopenalty[gpos] += cospenalty[consecutive - 2]
                                        slotpenalty[cosqueue[pos]] += cospenalty[consecutive - 2]
                                    slotnum = 0
                                    consecutive = 0
                            j += 1
                        else:
                            # Day changed, process penalties
                            if consecutive > 1:
                                for pos in range(slotnum):
                                    for gpos in chopos:
                                        if cosqueue[pos] == gen[gpos]:
                                            chopenalty[gpos] += cospenalty[consecutive - 2]
                                    slotpenalty[cosqueue[pos]] += cospenalty[consecutive - 2]

                            # Assign course penalty
                            if cosnum > 0:
                                for pos in range(cosnum):
                                    for gpos in chopos:
                                        if coqueue[pos] == gen[gpos]:
                                            chopenalty[gpos] += copenalty[cosnum - 1]
                                    slotpenalty[coqueue[pos]] += copenalty[cosnum - 1]

                            # Reset for new day
                            weekday = slot[j].day
                            consecutive = 0
                            slotnum = 0
                            cosnum = 0
                            cosqueue = []
                            coqueue = []

                    # Process penalties for last day
                    if consecutive > 1:
                        for pos in range(slotnum):
                            for gpos in chopos:
                                if cosqueue[pos] == gen[gpos]:
                                    chopenalty[gpos] += cospenalty[consecutive - 2]
                            slotpenalty[cosqueue[pos]] += cospenalty[consecutive - 2]

                    if cosnum > 0:
                        for pos in range(cosnum):
                            for gpos in chopos:
                                if coqueue[pos] == gen[gpos]:
                                    chopenalty[gpos] += copenalty[cosnum - 1]
                            slotpenalty[coqueue[pos]] += copenalty[cosnum - 1]

                if buf[z] == '$':
                    z += 1
    except IOError as e:
        print(f"[Violation] Error: {e}")
        exit(1)

def freeday(cos: List[Chromoson], slot: List[Period], gen: List[int],
            chopenalty: List[int], slotpenalty: List[int], daypenalty: List[int],
            maxslot: int, nslot: int):
    current = lecturerlist
    if current is None:
        print("\n\t[freeday] The lecturer list is empty!")
        exit(1)

    while current:
        node = current.hnext
        timeslot = [0] * nslot
        chopos = []

        # Collect all course slots for this lecturer
        while node:
            for pos in range(cos[node.code].frm, cos[node.code].to + 1):
                chopos.append(pos)
                timeslot[gen[pos]] = 1
            node = node.next

        # Count working days
        courseday = 0
        j = 0
        weekday = ""
        while j < nslot:
            if timeslot[j]:
                if not weekday:
                    weekday = slot[j].day
                    courseday += 1
                elif slot[j].day != weekday:
                    weekday = slot[j].day
                    courseday += 1
            j += 1

        # Assign penalties
        if courseday > 0:
            for gpos in chopos:
                chopenalty[gpos] += daypenalty[courseday - 1]
                slotpenalty[gen[gpos]] += daypenalty[courseday - 1]

        current = current.vnext

def Conslec(cos: List[Chromoson], slot: List[Period], gen: List[int],
            chopenalty: List[int], slotpenalty: List[int], lecpenalty: List[int], nslot: int):
    for code in range(TOTALCOURSE + 1):
        slotable = [0] * nslot
        queue = []

        # Mark lecture slots
        for pos in range(cos[code].frm, cos[code].frm + cos[code].lec):
            slotable[gen[pos]] += 1

        # Process by day
        weekday = slot[0].day
        num = 0
        fpos = 0

        for j in range(nslot):
            if slot[j].day == weekday:
                if slotable[j]:
                    queue.append(j)
                    num += slotable[j]
            else:
                # Day changed, assign penalties
                if num > 0:
                    for pos in range(cos[code].frm, cos[code].frm + cos[code].lec):
                        if gen[pos] in queue:
                            chopenalty[pos] += lecpenalty[num - 1]
                            slotpenalty[gen[pos]] += lecpenalty[num - 1]

                # Reset for new day
                weekday = slot[j].day
                num = 0
                queue = []
                if slotable[j]:
                    queue.append(j)
                    num += slotable[j]

        # Process last day
        if num > 0:
            for pos in range(cos[code].frm, cos[code].frm + cos[code].lec):
                if gen[pos] in queue:
                    chopenalty[pos] += lecpenalty[num - 1]
                    slotpenalty[gen[pos]] += lecpenalty[num - 1]

def Directed_mutation(cos: List[Chromoson], slot: List[Period],
                      clashtable: List[List[int]], parent: List[int], child: List[int],
                      copenalty: List[int], cospenalty: List[int], daypenalty: List[int],
                      lecpenalty: List[int], conslot: List[int], maxslot: int, nslot: int):
    for _ in range(6):  # Perform 6 directed mutations
        chopenalty = [0] * CHOLEN
        slotpenalty = [0] * nslot

        # Calculate penalties for current solution
        Violation(cos, slot, parent, chopenalty, slotpenalty, copenalty, cospenalty,
                  conslot, maxslot, nslot)
        freeday(cos, slot, parent, chopenalty, slotpenalty, daypenalty, maxslot, nslot)
        Conslec(cos, slot, parent, chopenalty, slotpenalty, lecpenalty, nslot)

        # Get the 6 worst genes (highest penalty)
        worstgen = [0] * 6
        temp_penalty = chopenalty.copy()
        compare(temp_penalty, worstgen, CHOLEN, 6, 1)

        # Get the 9 best timeslots (lowest penalty)
        bestslot = [0] * 9
        temp_slotpenalty = slotpenalty.copy()
        compare(temp_slotpenalty, bestslot, nslot, 9, 0)

        # Select one of the worst genes to mutate
        pos = random.randint(0, 5)

        # Determine the course code for this gene
        code = 0
        for c in range(TOTALCOURSE + 1):
            if worstgen[pos] >= cos[c].frm and worstgen[pos] <= cos[c].to:
                code = c
                break

        # Find conflicting slots
        clashslot = [0] * nslot
        for cpos in range(TOTALCOURSE + 1):
            if cpos == code or clashtable[cpos][code]:
                for gpos in range(cos[cpos].frm, cos[cpos].to + 1):
                    if pos == gpos:
                        continue
                    if (code != cpos) and (pos > (cos[code].frm + cos[code].lec - 1)) and \
                       (gpos > (cos[cpos].frm + cos[cpos].lec - 1)):
                        break
                    clashslot[parent[gpos]] = 1

        # Find non-conflicting best slots
        fpos = 0
        freeslot = []
        for j in range(9):
            if clashslot[bestslot[j]] == 0:
                freeslot.append(bestslot[j])
                fpos += 1

        # Mutate to one of the best available slots
        if fpos > 0:
            gpos = random.randint(0, fpos - 1)
        else:
            gpos = random.randint(0, 8)

        parent[worstgen[pos]] = bestslot[gpos]

    Ind_to_ind(child, parent)

def savecostime(outfile: str, cos: List[Chromoson], slot: List[Period],
                cho: List[int], size: int):
    try:
        with open(outfile, "w") as fd:
            code = 0
            cntl = True

            for i in range(size):
                if i > cos[code].to:
                    code += 1
                    cntl = True
                    fd.write("&\n")

                if i == cos[code].frm:
                    coursename = GetCourseName(code)
                    fd.write(f"{coursename}&")

                if cntl and i >= cos[code].frm + cos[code].lec:
                    cntl = False
                    fd.write("&")

                fd.write(f"{slot[cho[i]].day} {slot[cho[i]].time}$")

                if cos[code].tut == 0 and i == cos[code].to:
                    fd.write("&X")

            fd.write("&")
    except IOError as e:
        print(f"\n\t[savecostime] Error: {e}")
        exit(1)

def getfreeslot(clashtable: List[List[int]], cos: List[Chromoson],
                cho: List[int], code: int, col: int, nslot: int,
                freeslot: List[int], fpos: List[int]):
    clashslot = [0] * nslot

    # Check clash between courses
    for cpos in range(TOTALCOURSE + 1):
        if code == cpos:
            continue

        if clashtable[cpos][code]:
            for gpos in range(cos[cpos].frm, cos[cpos].to + 1):
                if (col > (cos[code].frm + cos[code].lec - 1)) and \
                   (gpos > (cos[cpos].frm + cos[cpos].lec - 1)):
                    break
                clashslot[cho[gpos]] = 1

    # Check clash between course items
    for gpos in range(cos[code].frm, cos[code].to + 1):
        if (cho[gpos] == -1) and (gpos == col):
            continue
        clashslot[cho[gpos]] = 1

    # Get free slots
    fpos[0] = 0
    for tpos in range(nslot):
        if clashslot[tpos] == 0:
            freeslot[fpos[0]] = tpos
            fpos[0] += 1

def sorting(freeslot: List[int], fitval: List[float], fpos: int):
    # Simple bubble sort
    for i in range(fpos):
        for j in range(i + 1, fpos):
            if fitval[i] < fitval[j]:
                # Swap both arrays
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

def CourseCons(cos: List[Chromoson], slot: List[Period], gen: List[int],
               numlec: List[int], nslot: int):
    for code in range(TOTALCOURSE + 1):
        slotable = [0] * nslot

        # Mark lecture slots
        for pos in range(cos[code].frm, cos[code].frm + cos[code].lec):
            slotable[gen[pos]] += 1

        # Process by day
        weekday = slot[0].day
        num = 0

        j = 0
        while j < nslot:
            if slot[j].day == weekday:
                if slotable[j]:
                    num += slotable[j]
                j += 1
            else:
                if num > 0:
                    numlec[num - 1] += 1
                weekday = slot[j].day
                num = 0

        if num > 0:
            numlec[num - 1] += 1

def StudentCons(cos: List[Chromoson], slot: List[Period], gen: List[int],
                conseperday: List[int], numcosperday: List[int], conslot: List[int],
                maxslot: int, nslot: int):
    try:
        with open("student.bin", "rb") as fd:
            buf = fd.read().decode('utf-8')

            z = 0
            while z < len(buf) and buf[z] != '&':
                course = []
                j = 0
                codestr = ""

                while buf[z] != '$' and z < len(buf) and buf[z] != '&':
                    if buf[z] != ' ':
                        codestr += buf[z]
                    else:
                        if codestr:
                            course.append(int(codestr))
                            codestr = ""
                    z += 1

                if codestr:  # Add last course
                    course.append(int(codestr))

                if course:
                    # Initialize tracking variables
                    weekday = slot[0].day
                    consecutive = 0
                    course_count = 0
                    timeslot = [0] * nslot

                    # Mark all slots for this student's courses
                    for coursecode in course:
                        for pos in range(cos[coursecode].frm, cos[coursecode].to + 1):
                            timeslot[gen[pos]] = 1

                    # Process time slots
                    j = 0
                    while j < nslot:
                        if slot[j].day == weekday:
                            if timeslot[j]:
                                # Check if consecutive with previous
                                if conslot[j] == 0 and j > 0 and timeslot[j-1]:
                                    if consecutive > 1:
                                        conseperday[consecutive - 2] += 1
                                    consecutive = 0

                                course_count += 1
                                consecutive += 1
                            else:
                                if consecutive > 1:
                                    conseperday[consecutive - 2] += 1
                                consecutive = 0
                            j += 1
                        else:
                            # Day changed
                            if consecutive > maxslot and course_count >= maxslot:
                                print("\n\t[StudentCons] Invalid number of slot!")
                                exit(1)

                            if consecutive > 1:
                                conseperday[consecutive - 2] += 1
                            if course_count > 0:
                                numcosperday[course_count - 1] += 1

                            weekday = slot[j].day
                            consecutive = 0
                            course_count = 0

                    # Process last day
                    if course_count > 0:
                        numcosperday[course_count - 1] += 1

                if buf[z] == '$':
                    z += 1
    except IOError as e:
        print(f"\n\t[StudentCons] Error: {e}")
        exit(1)

def StaffCons(cos: List[Chromoson], slot: List[Period], gen: List[int],
              numday: List[int], maxslot: int, nslot: int):
    current = lecturerlist
    if current is None:
        print("\n\t[StaffCons] The lecturer list is empty!")
        exit(1)

    while current:
        node = current.hnext
        timeslot = [0] * nslot

        # Mark all slots for this lecturer's courses
        while node:
            for pos in range(cos[node.code].frm, cos[node.code].to + 1):
                timeslot[gen[pos]] = 1
            node = node.next

        # Count working days
        courseday = 0
        j = 0
        weekday = ""

        while j < nslot:
            if timeslot[j]:
                if not weekday:
                    weekday = slot[j].day
                    courseday += 1
                elif slot[j].day != weekday:
                    weekday = slot[j].day
                    courseday += 1
            j += 1

        if courseday > 0:
            numday[courseday - 1] += 1

        current = current.vnext

def Fitness(cos: List[Chromoson], slot: List[Period], gen: List[int],
            cospenalty: List[int], copenalty: List[int], daypenalty: List[int],
            lecpenalty: List[int], tcosp: int, conslot: List[int],
            maxday: int, maxslot: int, s: int) -> float:
    numcosperday = [0] * maxslot
    conseperday = [0] * (maxslot - 1)
    numday = [0] * maxday
    numlec = [0] * tcosp
    total = 0

    StudentCons(cos, slot, gen, conseperday, numcosperday, conslot, maxslot, s)

    # Sum penalty for course spread constraint
    for i in range(maxslot):
        total += numcosperday[i] * copenalty[i]

    # Sum penalty for consecutive course constraint
    for i in range(maxslot - 1):
        total += conseperday[i] * cospenalty[i]

    CourseCons(cos, slot, gen, numlec, s)
    for i in range(tcosp):
        total += numlec[i] * lecpenalty[i]

    StaffCons(cos, slot, gen, numday, maxslot, s)
    for i in range(maxday):
        total += numday[i] * daypenalty[i]

    return 1.0 / (1.0 + float(total))

def final(Tablefile: str, lecsizefile: str, cos: List[Chromoson],
          slot: List[Period], cho: List[int], conslot: List[int], tcosp: int,
          maxslot: int, mday: int, nslot: int):
    try:
        # Remove existing file if it exists
        if os.path.exists(Tablefile):
            os.unlink(Tablefile)

        with open(Tablefile, "w") as fd:
            Lecsize = [0] * (TOTALCOURSE + 1)
            Tutsize = [0] * (TOTALCOURSE + 1)

            ReadLecTutsize(lecsizefile, Lecsize, Tutsize)

            tmpfile = f"{Tablefile}.tmp"
            fd.write(f"{tmpfile}\n\n")

            # Log statistics of soft constraints
            fitlog(fd, cos, slot, cho, conslot, tcosp, mday, maxslot, nslot)

            # Create timetable for all courses
            CreateTimeTable(fd, cos, slot, cho, maxslot, mday, Lecsize, Tutsize, "ALL COURSE TIMETABLE")

            fd.write("\n\t**********************************************************")
            fd.write("\n\t*            The timetable for every course              *")
            fd.write("\n\t**********************************************************\n\n")

            # Create separate timetable for each course
            for code in range(TOTALCOURSE + 1):
                CreateSlotTable(fd, cos, slot, cho, Lecsize, Tutsize, nslot, code)

            # Create timetable for each lecturer
            Lecturer_Timetable(fd, cos, slot, cho, Lecsize, Tutsize, nslot)
    except IOError as e:
        print(f"[final] Error: {e}")
        exit(1)

def Clash_Correct(cos: List[Chromoson], clashtable: List[List[int]],
                  fixslot: List[int], cho: List[List[int]], popsize: int, nslot: int):
    for row in range(popsize):
        col = 0
        code = 0
        first_flag = True
        tut_flag = True

        freeslot = [0] * nslot
        clashslot = [0] * nslot

        # Replace changed fix slots with original
        for pos in range(CHOLEN):
            if fixslot[pos] != -1:
                cho[row][pos] = fixslot[pos]

        while col < CHOLEN:
            cntl = False

            if col > cos[code].to:
                for tpos in range(nslot):
                    clashslot[tpos] = 0
                    freeslot[tpos] = 0
                cntl = True
                tut_flag = True
                code += 1

            if first_flag:
                cntl = True
                first_flag = False
                tut_flag = True

            if tut_flag and col > (cos[code].frm + cos[code].lec - 1):
                for tpos in range(nslot):
                    clashslot[tpos] = 0
                    freeslot[tpos] = 0
                cntl = True
                tut_flag = False

            # Check clash only for non-fixed slots
            if fixslot[col] == -1:
                if cntl:
                    # Check clash between courses
                    for cpos in range(TOTALCOURSE + 1):
                        if code == cpos:
                            continue

                        if clashtable[cpos][code]:
                            for gpos in range(cos[cpos].frm, cos[cpos].to + 1):
                                if (col > (cos[code].frm + cos[code].lec - 1)) and \
                                   (gpos > (cos[cpos].frm + cos[cpos].lec - 1)):
                                    break
                                clashslot[cho[row][gpos]] = 1

                # Check clash between course items
                for gpos in range(cos[code].frm, col):
                    clashslot[cho[row][gpos]] = 1

                # Get free slots
                fpos = 0
                for tpos in range(nslot):
                    if clashslot[tpos] == 0:
                        freeslot[fpos] = tpos
                        fpos += 1

                if clashslot[cho[row][col]] == 1:
                    if fpos > 0:
                        tpos = random.randint(0, fpos - 1)
                        cho[row][col] = freeslot[tpos]
                    else:
                        if not Backtracking(clashtable, cos, cho[row], fixslot, code, col, nslot):
                            print("No free slot")
                            exit(1)
                        else:
                            col += 1
                else:
                    col += 1
            else:
                col += 1

def Check_schedule_clash(clashtable: List[List[int]], cos: List[Chromoson],
                         slot: List[Period], fixslot: List[int], cho: List[int],
                         cospenalty: List[int], copenalty: List[int],
                         daypenalty: List[int], lecpenalty: List[int], tcosp: int,
                         conslot: List[int], maxday: int, maxslot: int,
                         nslot: int, popsize: int):
    col = 0
    code = 0
    first_flag = True
    tut_flag = True
    coursetype = "Lec"

    freeslot = [0] * nslot
    clashslot = [0] * nslot
    fitval = [0.0] * nslot

    while col < CHOLEN:
        cntl = False

        if col > cos[code].to:
            for tpos in range(nslot):
                clashslot[tpos] = 0
                freeslot[tpos] = 0
            coursetype = "Lec"
            cntl = True
            tut_flag = True
            code += 1

        if first_flag:
            cntl = True
            first_flag = False
            tut_flag = True

        if tut_flag and col > (cos[code].frm + cos[code].lec - 1):
            for tpos in range(nslot):
                clashslot[tpos] = 0
                freeslot[tpos] = 0
            coursetype = "Tut"
            cntl = True
            tut_flag = False

        # Check clash for fixed slots only
        if fixslot[col] != -1:
            if cntl:
                # Check clash between courses
                for cpos in range(TOTALCOURSE + 1):
                    if code == cpos:
                        continue

                    if clashtable[cpos][code]:
                        for gpos in range(cos[cpos].frm, cos[cpos].to + 1):
                            if (col > (cos[code].frm + cos[code].lec - 1)) and \
                               (gpos > (cos[cpos].frm + cos[cpos].lec - 1)):
                                break
                            clashslot[cho[gpos]] = 1

            # Check clash between course items
            for gpos in range(cos[code].frm, col):
                clashslot[cho[gpos]] = 1

            # Get free slots
            fpos = 0
            for tpos in range(nslot):
                if clashslot[tpos] == 0:
                    freeslot[fpos] = tpos
                    fpos += 1

            if clashslot[cho[col]] == 1:
                # Evaluate all possible free slots
                for tpos in range(fpos):
                    cho[col] = freeslot[tpos]
                    fitval[tpos] = Fitness(cos, slot, cho, cospenalty, copenalty,
                                          daypenalty, lecpenalty, tcosp, conslot, maxday, maxslot, nslot)

                # Sort by fitness
                sorting(freeslot, fitval, fpos)

                # Choose the best slot
                cho[col] = freeslot[0]
                course = GetCourseName(code)
                print(f"\n\tThe {course} ({coursetype}) is moved from {slot[fixslot[col]].day} {slot[fixslot[col]].time} to {slot[freeslot[0]].day} {slot[freeslot[0]].time}")
            else:
                col += 1
        else:
            col += 1

def GetMaxSlot(slot: List[Period], nslot: int, maxslot: List[int], maxday: List[int]):
    if nslot == 0:
        maxslot[0] = 0
        maxday[0] = 0
        return

    day = slot[0].