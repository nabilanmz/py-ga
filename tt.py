import csv
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
import random
import numpy as np
from deap import base, creator, tools, algorithms
from datetime import time, datetime, timedelta

# Initialize DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Constants
TIME_STEP = 30  # 30-minute intervals
IDEAL_GAP = 90  # 1.5 hour ideal gap between classes (minutes)
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


@dataclass
class Class:
    code: str
    course: str
    activity: str  # "Lecture" or "Tutorial"
    section: str
    days: str
    start_time: time
    end_time: time
    venue: str
    lecturer: str

    @property
    def duration(self) -> int:
        """Calculate duration in minutes"""
        start = datetime.combine(datetime.today(), self.start_time)
        end = datetime.combine(datetime.today(), self.end_time)
        return int((end - start).total_seconds() / 60)

    @property
    def unique_id(self) -> str:
        """Unique identifier for this class instance"""
        return f"{self.code}-{self.section}-{self.days}-{self.start_time}"


def load_classes_from_csv(filename: str) -> List[Class]:
    """Load classes from CSV file"""
    classes = []
    with open(filename, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                # Parse time (handle both "HH:MM AM/PM" and "HH:MM" formats)
                start_time = (
                    datetime.strptime(row["Start Time"], "%I:%M %p").time()
                    if "AM" in row["Start Time"] or "PM" in row["Start Time"]
                    else datetime.strptime(row["Start Time"], "%H:%M").time()
                )
                end_time = (
                    datetime.strptime(row["End Time"], "%I:%M %p").time()
                    if "AM" in row["End Time"] or "PM" in row["End Time"]
                    else datetime.strptime(row["End Time"], "%H:%M").time()
                )

                classes.append(
                    Class(
                        code=row["Code"],
                        course=row["Course"],
                        activity=row["Activity"],
                        section=row["Section"],
                        days=row["Days"],
                        start_time=start_time,
                        end_time=end_time,
                        venue=row["Venue"],
                        lecturer=row["Lecturer"] if row["Lecturer"] else "Not Assigned",
                    )
                )
            except ValueError as e:
                print(f"Skipping row due to error: {e}")
                continue
    return classes


def get_course_requirements(classes: List[Class]) -> Dict[str, Dict[str, int]]:
    """Generate course requirements based on available classes"""
    requirements = defaultdict(lambda: defaultdict(int))
    for cls in classes:
        requirements[cls.course][cls.activity] += 1

    # Convert to regular dict and set minimum requirements
    return {
        course: {
            "Lecture": min(1, counts.get("Lecture", 0)),  # At least 1 lecture section
            "Tutorial": min(
                1, counts.get("Tutorial", 0)
            ),  # At least 1 tutorial section
        }
        for course, counts in requirements.items()
    }


def group_classes_by_section(classes: List[Class]) -> Dict[str, Dict[str, List[Class]]]:
    """Group classes by course and section"""
    section_groups = defaultdict(lambda: defaultdict(list))
    for cls in classes:
        section_groups[cls.course][f"{cls.activity}_{cls.section}"].append(cls)
    return section_groups


@dataclass
class ScheduledClass:
    class_obj: Class
    day: str
    start_time: time
    end_time: time


class Timetable:
    def __init__(self):
        self.schedule = {day: [] for day in DAYS}
        self.scheduled_classes = []
        # Track which sections have been scheduled for each course
        self.scheduled_sections = defaultdict(set)

    def add_class(self, scheduled_class: ScheduledClass) -> bool:
        class_obj = scheduled_class.class_obj
        day = scheduled_class.day
        start = scheduled_class.start_time
        end = scheduled_class.end_time

        # Check for overlapping classes
        for existing in self.schedule[day]:
            if not (end <= existing.start_time or start >= existing.end_time):
                return False

        # Check lecturer availability (skip if lecturer not assigned)
        lecturer = class_obj.lecturer
        if lecturer != "Not Assigned":
            for existing in self.scheduled_classes:
                if existing.class_obj.lecturer == lecturer:
                    if not (end <= existing.start_time or start >= existing.end_time):
                        return False

        # If all checks passed, add the class
        self.schedule[day].append(scheduled_class)
        self.scheduled_classes.append(scheduled_class)
        return True

    def add_section(self, section_classes: List[Class]) -> bool:
        """Add all classes in a section (must add all or none)"""
        # First check if we can add all classes
        temp_schedule = {day: list(classes) for day, classes in self.schedule.items()}
        temp_scheduled_classes = list(self.scheduled_classes)

        for cls in section_classes:
            sc = ScheduledClass(
                class_obj=cls,
                day=cls.days,
                start_time=cls.start_time,
                end_time=cls.end_time,
            )

            # Check for overlapping classes
            for existing in temp_schedule[cls.days]:
                if not (
                    sc.end_time <= existing.start_time
                    or sc.start_time >= existing.end_time
                ):
                    return False

            # Check lecturer availability
            if cls.lecturer != "Not Assigned":
                for existing in temp_scheduled_classes:
                    if existing.class_obj.lecturer == cls.lecturer:
                        if not (
                            sc.end_time <= existing.start_time
                            or sc.start_time >= existing.end_time
                        ):
                            return False

            temp_schedule[cls.days].append(sc)
            temp_scheduled_classes.append(sc)

        # If we get here, all classes can be added
        for cls in section_classes:
            sc = ScheduledClass(
                class_obj=cls,
                day=cls.days,
                start_time=cls.start_time,
                end_time=cls.end_time,
            )
            self.add_class(sc)

        # Mark this section as scheduled
        first_class = section_classes[0]
        self.scheduled_sections[first_class.course].add(
            f"{first_class.activity}_{first_class.section}"
        )
        return True

    def get_utilized_days(self) -> int:
        return sum(1 for day in DAYS if self.schedule[day])

    def get_class_gaps(self) -> Dict[str, List[float]]:
        subject_times = defaultdict(list)
        for sc in self.scheduled_classes:
            subject_times[sc.class_obj.course].append(
                (sc.day, datetime.combine(datetime.today(), sc.start_time))
            )

        gaps = defaultdict(list)
        for subject, times in subject_times.items():
            times.sort(key=lambda x: x[1])
            for i in range(1, len(times)):
                if times[i][0] == times[i - 1][0]:  # Same day
                    gap = (times[i][1] - times[i - 1][1]).total_seconds() / 60
                    gaps[subject].append(gap)
        return gaps


class TimetableGenerator:
    def __init__(self, classes: List[Class], user_preferences: dict):
        self.classes = classes
        self.user_preferences = user_preferences
        self.course_requirements = get_course_requirements(classes)
        self.section_groups = group_classes_by_section(classes)
        self.setup_deap()

    def setup_deap(self):
        self.toolbox = base.Toolbox()

        # Create a list of all section groups for preferred courses
        self.valid_sections = []
        self.section_index_map = {}
        index = 0

        for course in self.user_preferences["courses"]:
            if course in self.section_groups:
                for section_key, section_classes in self.section_groups[course].items():
                    self.valid_sections.append((course, section_key, section_classes))
                    self.section_index_map[(course, section_key)] = index
                    index += 1

        self.toolbox.register(
            "attr_bool", random.randint, 0, len(self.valid_sections) - 1
        )
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self.toolbox.attr_bool,
            n=len(self.valid_sections),
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register(
            "mutate",
            tools.mutUniformInt,
            low=0,
            up=len(self.valid_sections) - 1,
            indpb=0.1,
        )
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def evaluate(self, individual: Tuple) -> Tuple[float]:
        timetable = Timetable()
        score = 0

        # Track which sections we're trying to schedule
        selected_sections = set()
        for section_idx in individual:
            course, section_key, section_classes = self.valid_sections[section_idx]
            selected_sections.add((course, section_key))

        # Try to schedule each selected section
        for course, section_key in selected_sections:
            _, _, section_classes = self.valid_sections[
                self.section_index_map[(course, section_key)]
            ]

            # Check if this section matches preferred days/times
            matches_prefs = all(
                cls.days in self.user_preferences["preferred_days"]
                and self.user_preferences["preferred_start"]
                <= cls.start_time
                <= self.user_preferences["preferred_end"]
                for cls in section_classes
            )

            if timetable.add_section(section_classes):
                score += 10 * len(section_classes)  # Base score for scheduling

                # Bonus for preferred time
                if matches_prefs:
                    score += 5 * len(section_classes)
            else:
                score -= 5 * len(section_classes)  # Penalty for not scheduling

        # Check course requirements
        course_activities = defaultdict(set)
        for sc in timetable.scheduled_classes:
            course_activities[sc.class_obj.course].add(sc.class_obj.activity)

        # Reward for meeting requirements
        for course in self.user_preferences["courses"]:
            if course in self.course_requirements:
                for activity, count in self.course_requirements[course].items():
                    if activity in course_activities[course]:
                        score += 20
                    else:
                        score -= 10

        # Reward for preferred days
        utilized_days = timetable.get_utilized_days()
        preferred_days_count = len(
            [
                day
                for day in timetable.schedule
                if day in self.user_preferences["preferred_days"]
                and timetable.schedule[day]
            ]
        )
        score += preferred_days_count * 5

        return (max(score, 0.1),)

    def run(self, generations=100) -> Timetable:
        pop = self.toolbox.population(n=50)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)

        algorithms.eaSimple(
            pop,
            self.toolbox,
            cxpb=0.7,
            mutpb=0.2,
            ngen=generations,
            stats=stats,
            halloffame=hof,
            verbose=True,
        )

        # Build the best timetable
        best_ind = hof[0]
        timetable = Timetable()
        selected_sections = set()

        # First collect all selected sections
        for section_idx in best_ind:
            course, section_key, _ = self.valid_sections[section_idx]
            selected_sections.add((course, section_key))

        # Try to add sections in a way that maximizes the score
        for course, section_key in selected_sections:
            _, _, section_classes = self.valid_sections[
                self.section_index_map[(course, section_key)]
            ]
            timetable.add_section(section_classes)

        return timetable


def get_user_preferences(classes: List[Class]) -> dict:
    """Get user preferences with validation"""
    # Get unique courses
    courses = sorted({cls.course for cls in classes})
    print("\nAvailable Courses:")
    for i, course in enumerate(courses, 1):
        print(f"{i}. {course}")

    while True:
        try:
            selections = (
                input("\nEnter preferred course numbers (comma separated): ")
                .strip()
                .split(",")
            )
            selected_courses = [
                courses[int(sel) - 1] for sel in selections if sel.strip()
            ]
            if not selected_courses:
                print("Please select at least one course")
                continue
            break
        except (ValueError, IndexError):
            print("Invalid selection. Please enter numbers from the list.")

    print("\nAvailable Days:", DAYS)
    while True:
        preferred_days = (
            input("Enter preferred days (comma separated): ").strip().split(",")
        )
        preferred_days = [day.strip().capitalize() for day in preferred_days]
        # Validate days
        invalid_days = [day for day in preferred_days if day not in DAYS]
        if invalid_days:
            print(f"Invalid days: {invalid_days}. Please choose from {DAYS}")
        else:
            break

    print("\nPreferred time range (24-hour format)")
    while True:
        try:
            start = input("Earliest preferred start time (e.g., 09:00): ").strip()
            end = input("Latest preferred end time (e.g., 16:00): ").strip()
            preferred_start = datetime.strptime(start, "%H:%M").time()
            preferred_end = datetime.strptime(end, "%H:%M").time()
            if preferred_start >= preferred_end:
                print("End time must be after start time")
                continue
            break
        except ValueError:
            print("Invalid time format. Please use HH:MM (24-hour format)")

    return {
        "courses": selected_courses,
        "preferred_days": preferred_days,
        "preferred_start": preferred_start,
        "preferred_end": preferred_end,
    }


def print_timetable(timetable: Timetable):
    """Print the timetable in readable format"""
    print("\n=== Optimized Timetable ===")
    for day in DAYS:
        print(f"\n{day}:")
        if not timetable.schedule[day]:
            print("No classes")
            continue

        for sc in sorted(timetable.schedule[day], key=lambda x: x.start_time):
            print(
                f"{sc.start_time.strftime('%H:%M')}-{sc.end_time.strftime('%H:%M')}: "
                f"{sc.class_obj.course} {sc.class_obj.activity} (Section {sc.class_obj.section}) "
                f"at {sc.class_obj.venue} with {sc.class_obj.lecturer}"
            )


def print_requirements_check(timetable: Timetable, course_requirements: dict):
    """Verify all requirements are met"""
    print("\n=== Requirements Check ===")
    course_activities = defaultdict(set)
    for sc in timetable.scheduled_classes:
        course_activities[sc.class_obj.course].add(sc.class_obj.activity)

    for course, reqs in course_requirements.items():
        if course not in course_activities:
            continue

        print(f"\n{course}:")
        for activity, count in reqs.items():
            status = "✓" if activity in course_activities[course] else f"✗ (missing)"
            print(f"  {activity}: {status}")


def print_section_enrollment(timetable: Timetable):
    """Show which sections are enrolled"""
    print("\n=== Section Enrollment ===")
    enrolled_sections = defaultdict(set)
    for sc in timetable.scheduled_classes:
        enrolled_sections[sc.class_obj.course].add(
            f"{sc.class_obj.activity} {sc.class_obj.section}"
        )

    for course, sections in enrolled_sections.items():
        print(f"\n{course}:")
        for section in sorted(sections):
            print(f"  - {section}")


def main():
    print("=== University Timetable Generator ===")

    # Load classes from CSV
    classes = load_classes_from_csv("classes.csv")
    print(f"Loaded {len(classes)} classes from CSV")

    # Get user preferences
    user_prefs = get_user_preferences(classes)

    print("\nGenerating timetable based on your preferences...")
    generator = TimetableGenerator(classes, user_prefs)
    best_timetable = generator.run(
        generations=50
    )  # Reduced generations for faster testing

    print_timetable(best_timetable)
    print_requirements_check(best_timetable, generator.course_requirements)
    print_section_enrollment(best_timetable)

    # Statistics
    print("\n=== Schedule Statistics ===")
    print(f"Days used: {best_timetable.get_utilized_days()} of {len(DAYS)}")
    print(
        f"Preferred days used: {len([d for d in user_prefs['preferred_days'] if best_timetable.schedule[d]])}"
    )


if __name__ == "__main__":
    main()
