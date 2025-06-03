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
        self.scheduled_sections = defaultdict(
            set
        )  # Track scheduled sections per course

    def can_add_section(self, section_classes: List[Class]) -> bool:
        """Check if we can add all classes in this section"""
        first_class = section_classes[0]
        activity_type = first_class.activity

        # Check if we already have a section of this type for the course
        for scheduled_section in self.scheduled_sections[first_class.course]:
            if scheduled_section.startswith(activity_type):
                return False  # Already have a section of this type

        # Check time and lecturer conflicts
        for cls in section_classes:
            # Check time conflicts
            for existing in self.schedule[cls.days]:
                if not (
                    cls.end_time <= existing.start_time
                    or cls.start_time >= existing.end_time
                ):
                    return False

            # Check lecturer conflicts (if lecturer is assigned)
            if cls.lecturer != "Not Assigned":
                for existing in self.scheduled_classes:
                    if existing.class_obj.lecturer == cls.lecturer:
                        if not (
                            cls.end_time <= existing.start_time
                            or cls.start_time >= existing.end_time
                        ):
                            return False

        return True

    def add_section(self, section_classes: List[Class]) -> bool:
        """Add all classes in a section"""
        if not self.can_add_section(section_classes):
            return False

        # Add all classes in the section
        for cls in section_classes:
            sc = ScheduledClass(
                class_obj=cls,
                day=cls.days,
                start_time=cls.start_time,
                end_time=cls.end_time,
            )
            self.schedule[cls.days].append(sc)
            self.scheduled_classes.append(sc)

        # Record that we've scheduled this section
        first_class = section_classes[0]
        self.scheduled_sections[first_class.course].add(
            f"{first_class.activity}_{first_class.section}"
        )
        return True

    def get_utilized_days(self) -> int:
        return sum(1 for day in DAYS if self.schedule[day])


class TimetableGenerator:
    def __init__(self, classes: List[Class], user_preferences: dict):
        self.classes = classes
        self.user_preferences = user_preferences
        self.section_groups = group_classes_by_section(classes)
        self.setup_deap()

    def setup_deap(self):
        self.toolbox = base.Toolbox()

        # For each course, create lists of available lecture and tutorial sections
        self.course_sections = defaultdict(lambda: defaultdict(list))
        for course in self.user_preferences["courses"]:
            if course in self.section_groups:
                for section_key, section_classes in self.section_groups[course].items():
                    activity = section_classes[0].activity
                    self.course_sections[course][activity].append(section_classes)

        # Create a list of all possible section choices
        self.section_choices = []
        self.section_info = []  # Stores (course, activity, section_classes)

        for course in self.user_preferences["courses"]:
            for activity in ["Lecture", "Tutorial"]:
                if activity in self.course_sections[course]:
                    for section_classes in self.course_sections[course][activity]:
                        self.section_choices.append((course, activity, section_classes))
                        self.section_info.append((course, activity, section_classes))

        # Each gene represents whether to include a section (0 or 1)
        self.toolbox.register("attr_bool", random.randint, 0, 1)
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self.toolbox.attr_bool,
            n=len(self.section_choices),
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def evaluate(self, individual: List[int]) -> Tuple[float]:
        timetable = Timetable()
        score = 0

        # Track which sections we've selected for each course and activity type
        selected_sections = defaultdict(lambda: defaultdict(list))
        for idx, selected in enumerate(individual):
            if selected:
                course, activity, section_classes = self.section_info[idx]
                selected_sections[course][activity].append(section_classes)

        # Validate and score each selected section
        for course in self.user_preferences["courses"]:
            # Ensure we have exactly one lecture and one tutorial (if available)
            lectures = selected_sections[course]["Lecture"]
            tutorials = selected_sections[course]["Tutorial"]

            has_lecture = (
                len(lectures) > 0 and "Lecture" in self.course_sections[course]
            )
            has_tutorial = (
                len(tutorials) > 0 and "Tutorial" in self.course_sections[course]
            )

            # Penalize if missing required sections
            if "Lecture" in self.course_sections[course] and not has_lecture:
                score -= 1000
            if "Tutorial" in self.course_sections[course] and not has_tutorial:
                score -= 1000

            # Penalize if we have more than one of any type
            if len(lectures) > 1 or len(tutorials) > 1:
                score -= 1000

            # Try to add each valid section
            for activity in ["Lecture", "Tutorial"]:
                for section_classes in selected_sections[course][activity]:
                    # Check if section matches preferences
                    matches_prefs = all(
                        cls.days in self.user_preferences["preferred_days"]
                        and self.user_preferences["preferred_start"]
                        <= cls.start_time
                        <= self.user_preferences["preferred_end"]
                        for cls in section_classes
                    )

                    if timetable.add_section(section_classes):
                        score += 10 * len(section_classes)  # Base score
                        if matches_prefs:
                            score += 5 * len(section_classes)  # Preference bonus
                    else:
                        score -= 5 * len(section_classes)  # Penalty for conflict

        # Additional reward for having both lecture and tutorial for each course
        for course in self.user_preferences["courses"]:
            has_lecture = any(
                s.startswith("Lecture") for s in timetable.scheduled_sections[course]
            )
            has_tutorial = any(
                s.startswith("Tutorial") for s in timetable.scheduled_sections[course]
            )

            if has_lecture and has_tutorial:
                score += 20
            elif has_lecture or has_tutorial:
                score += 10

        # Reward for preferred days
        preferred_days_count = len(
            [
                d
                for d in self.user_preferences["preferred_days"]
                if any(sc.day == d for sc in timetable.scheduled_classes)
            ]
        )
        score += preferred_days_count * 3

        return (max(score, 1),)  # Ensure minimum score of 1

    def run(self, generations=100) -> Timetable:
        pop = self.toolbox.population(n=200)  # Larger population for better exploration
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
        best_timetable = Timetable()

        # First pass: Add required sections (one lecture and one tutorial per course)
        for course in self.user_preferences["courses"]:
            # Add one lecture section if available
            if "Lecture" in self.course_sections[course]:
                for idx, (c, a, _) in enumerate(self.section_info):
                    if c == course and a == "Lecture" and best_ind[idx]:
                        section_classes = self.section_info[idx][2]
                        best_timetable.add_section(section_classes)
                        break

            # Add one tutorial section if available
            if "Tutorial" in self.course_sections[course]:
                for idx, (c, a, _) in enumerate(self.section_info):
                    if c == course and a == "Tutorial" and best_ind[idx]:
                        section_classes = self.section_info[idx][2]
                        best_timetable.add_section(section_classes)
                        break

        return best_timetable


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

        # Group by course and section for better display
        day_classes = defaultdict(list)
        for sc in timetable.schedule[day]:
            key = f"{sc.class_obj.course} - {sc.class_obj.activity} {sc.class_obj.section}"
            day_classes[key].append(sc)

        for section, classes in sorted(day_classes.items()):
            classes_sorted = sorted(classes, key=lambda x: x.start_time)
            print(f"\n{section}:")
            for sc in classes_sorted:
                print(
                    f"  {sc.start_time.strftime('%H:%M')}-{sc.end_time.strftime('%H:%M')} "
                    f"at {sc.class_obj.venue} with {sc.class_obj.lecturer}"
                )


def print_section_summary(timetable: Timetable):
    """Show which sections were selected"""
    print("\n=== Selected Sections ===")
    sections_by_course = defaultdict(lambda: defaultdict(list))
    for sc in timetable.scheduled_classes:
        sections_by_course[sc.class_obj.course][sc.class_obj.activity].append(
            sc.class_obj.section
        )

    for course, activities in sections_by_course.items():
        print(f"\n{course}:")
        for activity, sections in activities.items():
            unique_sections = sorted(set(sections))
            print(f"  {activity}: {', '.join(unique_sections)}")


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
        generations=100
    )  # Increased generations for better results

    print_timetable(best_timetable)
    print_section_summary(best_timetable)

    # Statistics
    print("\n=== Schedule Statistics ===")
    print(f"Days used: {best_timetable.get_utilized_days()} of {len(DAYS)}")

    preferred_days_used = len(
        [
            d
            for d in user_prefs["preferred_days"]
            if any(sc.day == d for sc in best_timetable.scheduled_classes)
        ]
    )
    print(
        f"Preferred days used: {preferred_days_used} of {len(user_prefs['preferred_days'])}"
    )


if __name__ == "__main__":
    main()
