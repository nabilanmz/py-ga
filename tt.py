import csv
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
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
    def possible_times(self) -> List[Dict[str, List[time]]]:
        """Format for genetic algorithm compatibility"""
        return [{self.days: [self.start_time]}]


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
            "Lecture": min(
                1, counts.get("Lecture", 0)
            ),  # At least 1 lecture if available
            "Tutorial": min(
                1, counts.get("Tutorial", 0)
            ),  # At least 1 tutorial if available
        }
        for course, counts in requirements.items()
    }


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

    def add_class(self, scheduled_class: ScheduledClass) -> bool:
        day = scheduled_class.day
        start = scheduled_class.start_time
        end = scheduled_class.end_time

        # Check for overlapping classes
        for existing in self.schedule[day]:
            if not (end <= existing.start_time or start >= existing.end_time):
                return False

        # Check lecturer availability (skip if lecturer not assigned)
        lecturer = scheduled_class.class_obj.lecturer
        if lecturer != "Not Assigned":
            for existing in self.scheduled_classes:
                if existing.class_obj.lecturer == lecturer:
                    if not (end <= existing.start_time or start >= existing.end_time):
                        return False

        self.schedule[day].append(scheduled_class)
        self.scheduled_classes.append(scheduled_class)
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
        self.setup_deap()

    def setup_deap(self):
        self.toolbox = base.Toolbox()

        # Only consider classes for preferred courses
        valid_classes = [
            i
            for i, cls in enumerate(self.classes)
            if cls.course in self.user_preferences["courses"]
        ]

        self.toolbox.register("attr_bool", lambda: random.choice(valid_classes))
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self.toolbox.attr_bool,
            n=len(self.classes),
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register(
            "mutate",
            tools.mutUniformInt,
            low=min(valid_classes),
            up=max(valid_classes),
            indpb=0.1,
        )
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def evaluate(self, individual: Tuple) -> Tuple[float]:
        timetable = Timetable()
        selected_classes = [self.classes[i] for i in individual]
        score = 0

        # Try to schedule each selected class
        for class_obj in selected_classes:
            # Skip if not a preferred course
            if class_obj.course not in self.user_preferences["courses"]:
                continue

            # Skip if not a preferred day
            if class_obj.days not in self.user_preferences["preferred_days"]:
                continue

            # Check if time matches user preferences
            preferred_start = self.user_preferences["preferred_start"]
            preferred_end = self.user_preferences["preferred_end"]
            if not (preferred_start <= class_obj.start_time <= preferred_end):
                continue

            sc = ScheduledClass(
                class_obj=class_obj,
                day=class_obj.days,
                start_time=class_obj.start_time,
                end_time=class_obj.end_time,
            )

            if timetable.add_class(sc):
                score += 10  # Base score for scheduling

                # Bonus for preferred time
                if preferred_start <= class_obj.start_time <= preferred_end:
                    score += 5
            else:
                score -= 5  # Penalty for not scheduling

        # Check course requirements
        course_counts = defaultdict(lambda: defaultdict(int))
        for sc in timetable.scheduled_classes:
            course_counts[sc.class_obj.course][sc.class_obj.activity] += 1

        # Reward for meeting requirements
        for course in self.user_preferences["courses"]:
            if course in self.course_requirements:
                for activity, count in self.course_requirements[course].items():
                    if course_counts[course][activity] >= count:
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
        for class_idx in best_ind:
            class_obj = self.classes[class_idx]
            # Only add if it matches preferences
            if (
                class_obj.course in self.user_preferences["courses"]
                and class_obj.days in self.user_preferences["preferred_days"]
                and self.user_preferences["preferred_start"]
                <= class_obj.start_time
                <= self.user_preferences["preferred_end"]
            ):

                sc = ScheduledClass(
                    class_obj=class_obj,
                    day=class_obj.days,
                    start_time=class_obj.start_time,
                    end_time=class_obj.end_time,
                )
                timetable.add_class(sc)

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
                f"{sc.class_obj.course} {sc.class_obj.activity} ({sc.class_obj.code}) "
                f"at {sc.class_obj.venue} with {sc.class_obj.lecturer}"
            )


def print_requirements_check(timetable: Timetable, course_requirements: dict):
    """Verify all requirements are met"""
    print("\n=== Requirements Check ===")
    course_counts = defaultdict(lambda: defaultdict(int))
    for sc in timetable.scheduled_classes:
        course_counts[sc.class_obj.course][sc.class_obj.activity] += 1

    for course, reqs in course_requirements.items():
        if course not in {sc.class_obj.course for sc in timetable.scheduled_classes}:
            continue

        print(f"\n{course}:")
        for activity, count in reqs.items():
            actual = course_counts[course][activity]
            status = "✓" if actual >= count else f"✗ (needs {count - actual} more)"
            print(f"  {activity}: {actual}/{count} {status}")


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

    # Statistics
    print("\n=== Schedule Statistics ===")
    print(f"Days used: {best_timetable.get_utilized_days()} of {len(DAYS)}")
    print(
        f"Preferred days used: {len([d for d in user_prefs['preferred_days'] if best_timetable.schedule[d]])}"
    )


if __name__ == "__main__":
    main()
