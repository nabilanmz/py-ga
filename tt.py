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
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
MAX_CONSECUTIVE_CLASSES = 2  # Maximum preferred consecutive classes per day
IDEAL_GAP = timedelta(hours=1)  # 1 hour gap is ideal
MAX_GAP = timedelta(hours=2)  # More than 2 hours gap is not preferred


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
    def time_tuple(self) -> Tuple[datetime, datetime]:
        """Return start and end as datetime objects for comparison"""
        today = datetime.today()
        return (
            datetime.combine(today, self.start_time),
            datetime.combine(today, self.end_time),
        )


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

    def can_add_section(self, section_classes: List[Class]) -> bool:
        """Check if we can add all classes in this section without clashes."""
        # This function now only needs to check for time clashes, as the GA
        # structure handles the one-lecture-per-course logic.
        for cls in section_classes:
            # Check time conflicts
            for existing in self.schedule[cls.days]:
                # A clash occurs if the new class starts before the existing one ends
                # AND the new class ends after the existing one starts.
                if (
                    cls.start_time < existing.end_time
                    and cls.end_time > existing.start_time
                ):
                    return False
        return True

    def add_section(self, section_classes: List[Class]):
        """Add all classes in a section. Assumes can_add_section was checked."""
        for cls in section_classes:
            sc = ScheduledClass(
                class_obj=cls,
                day=cls.days,
                start_time=cls.start_time,
                end_time=cls.end_time,
            )
            self.schedule[cls.days].append(sc)
            # Sort the day's schedule by start time after adding
            self.schedule[cls.days].sort(key=lambda x: x.start_time)
            self.scheduled_classes.append(sc)

    # All other Timetable methods (get_utilized_days, get_consecutive_days_score, etc.)
    # remain the same. The meets_requirements method is no longer needed.

    def get_utilized_days(self) -> int:
        return sum(1 for day in DAYS if self.schedule[day])

    def get_consecutive_days_score(self) -> float:
        """Calculate score based on consecutive days used"""
        days_used = [day for day in DAYS if self.schedule[day]]
        if not days_used:
            return 0

        # Calculate longest streak of consecutive days
        max_streak = current_streak = 1
        for i in range(1, len(days_used)):
            if DAYS.index(days_used[i]) == DAYS.index(days_used[i - 1]) + 1:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1

        return max_streak / len(DAYS)  # Normalized score

    def get_day_gaps_score(self, day: str) -> float:
        """Calculate score based on gaps between classes on a single day"""
        day_classes = sorted(self.schedule[day], key=lambda x: x.start_time)
        if len(day_classes) < 2:
            return 1.0  # No gaps if only one class

        total_gap_score = 0
        consecutive_count = 1

        for i in range(1, len(day_classes)):
            prev_end = datetime.combine(datetime.today(), day_classes[i - 1].end_time)
            curr_start = datetime.combine(datetime.today(), day_classes[i].start_time)
            gap = curr_start - prev_end

            if gap <= timedelta(0):
                consecutive_count += 1
                continue  # No gap or overlap
            elif gap <= IDEAL_GAP:
                total_gap_score += 1.0  # Perfect gap
            elif gap <= MAX_GAP:
                total_gap_score += 0.5  # Acceptable gap
            else:
                total_gap_score += 0.1  # Too long gap

            consecutive_count = 1

        # Penalize too many consecutive classes
        if consecutive_count > MAX_CONSECUTIVE_CLASSES:
            total_gap_score *= 0.7  # Reduce score for too many consecutive classes

        return total_gap_score / (len(day_classes) - 1) if len(day_classes) > 1 else 1.0

    def get_scheduled_courses(self) -> Set[str]:
        return {sc.class_obj.course for sc in self.scheduled_classes}

    def meets_requirements(self, required_courses: List[str]) -> bool:
        """Check if all required courses are properly scheduled"""
        for course in required_courses:
            has_lecture = any(
                s.startswith("Lecture") for s in self.scheduled_sections[course]
            )
            has_tutorial = any(
                s.startswith("Tutorial") for s in self.scheduled_sections[course]
            )

            # Check if course has at least one lecture (if available)
            if any(
                s.startswith("Lecture") for s in self.section_groups.get(course, {})
            ):
                if not has_lecture:
                    return False

            # Check if course has at least one tutorial (if available)
            if any(
                s.startswith("Tutorial") for s in self.section_groups.get(course, {})
            ):
                if not has_tutorial:
                    return False

        return True


class TimetableGenerator:
    def __init__(self, classes: List[Class], user_preferences: dict):
        self.classes = classes
        self.user_preferences = user_preferences
        self.section_groups = group_classes_by_section(classes)
        self.gene_map = []  # ### NEW ###: This will map genes to actual sections
        self.setup_deap()

    def setup_deap(self):
        self.toolbox = base.Toolbox()

        # ### NEW ###: Create a map of genes to choices
        # Each gene in the individual will correspond to an entry in this map.
        gene_upper_bounds = []

        for course in self.user_preferences["courses"]:
            if course in self.section_groups:
                # Get all lecture sections for this course
                lectures = [
                    sc
                    for sk, sc in self.section_groups[course].items()
                    if sk.startswith("Lecture")
                ]
                if lectures:
                    self.gene_map.append(("Lecture", course, lectures))
                    gene_upper_bounds.append(len(lectures) - 1)

                # Get all tutorial sections for this course
                tutorials = [
                    sc
                    for sk, sc in self.section_groups[course].items()
                    if sk.startswith("Tutorial")
                ]
                if tutorials:
                    self.gene_map.append(("Tutorial", course, tutorials))
                    gene_upper_bounds.append(len(tutorials) - 1)

        if not self.gene_map:
            raise ValueError("No valid sections found for the selected courses.")

        # ### CHANGED ###: An individual is a list of integers (choices)
        # Each gene is an integer from 0 to the number of available sections for that slot.
        self.toolbox.register(
            "indices",
            lambda bounds: [random.randint(0, b) for b in bounds],
            gene_upper_bounds,
        )
        self.toolbox.register(
            "individual", tools.initIterate, creator.Individual, self.toolbox.indices
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        # Mutation for integer-based individuals
        self.toolbox.register(
            "mutate",
            tools.mutUniformInt,
            low=[0] * len(gene_upper_bounds),
            up=gene_upper_bounds,
            indpb=0.1,
        )
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def evaluate(self, individual: List[int]) -> Tuple[float,]:
        # ### THIS IS THE METHOD TO REPLACE ###
        timetable = Timetable()
        MAX_CONSECUTIVE_CLASSES = 2  # You can adjust this preference

        # Step 1: Check for HARD constraints (clashes). If any fail, fitness is 0.
        for i, choice_index in enumerate(individual):
            activity, course, sections = self.gene_map[i]
            chosen_section = sections[choice_index]

            if not timetable.can_add_section(chosen_section):
                return (0,)  # Clash detected. This is an invalid timetable.

            # No clash, so add it to our temporary timetable for further checks
            timetable.add_section(chosen_section)

        # Step 2: If we reach here, the timetable is valid.
        # Now, score it based on SOFT constraints (preferences).
        score = 10000.0  # High base score for being a valid solution

        # Penalty for using more days
        days_used = timetable.get_utilized_days()
        score -= days_used * 500

        # Score for gaps (using your existing logic, it's still good for rewarding breaks)
        total_gap_score = 0
        utilized_days = [day for day in DAYS if timetable.schedule[day]]
        if utilized_days:
            for day in utilized_days:
                # We use the existing gap score function here
                total_gap_score += timetable.get_day_gaps_score(day)
            score += (total_gap_score / len(utilized_days)) * 1000

        # Bonus for using preferred days
        for sc in timetable.scheduled_classes:
            if sc.day in self.user_preferences["preferred_days"]:
                score += 100

        # Bonus for classes within preferred time range
        for sc in timetable.scheduled_classes:
            if (
                self.user_preferences["preferred_start"]
                <= sc.start_time
                <= self.user_preferences["preferred_end"]
            ):
                score += 50

        # ### NEW & IMPROVED: Step 3 - Apply a heavy penalty for too many consecutive classes ###
        for day in DAYS:
            day_classes = timetable.schedule[day]
            if len(day_classes) <= MAX_CONSECUTIVE_CLASSES:
                continue  # No penalty needed if the total classes for the day is within the limit

            consecutive_streak = 1
            for i in range(1, len(day_classes)):
                prev_class_end = day_classes[i - 1].end_time
                curr_class_start = day_classes[i].start_time

                # Check if classes are back-to-back (allowing for a small, e.g., 15-min, travel gap)
                prev_end_dt = datetime.combine(datetime.today(), prev_class_end)
                curr_start_dt = datetime.combine(datetime.today(), curr_class_start)

                if (curr_start_dt - prev_end_dt) <= timedelta(minutes=15):
                    consecutive_streak += 1
                else:
                    # The streak is broken, check if the previous streak was too long
                    if consecutive_streak > MAX_CONSECUTIVE_CLASSES:
                        # Apply penalty for each class over the limit
                        over_limit = consecutive_streak - MAX_CONSECUTIVE_CLASSES
                        # The penalty should be significant enough to matter
                        penalty = over_limit * 750  # Heavy penalty per extra class
                        score -= penalty
                    consecutive_streak = 1  # Reset streak

            # Final check for the last streak of the day
            if consecutive_streak > MAX_CONSECUTIVE_CLASSES:
                over_limit = consecutive_streak - MAX_CONSECUTIVE_CLASSES
                penalty = over_limit * 750
                score -= penalty

        return (score,)

    def run(self, generations=100, pop_size=300) -> Optional[Timetable]:
        # ### CHANGED ###: Logic to handle cases where no solution is found.
        if not self.gene_map:
            print(
                "\nError: No sections available for the selected courses. Cannot generate a timetable."
            )
            return None

        pop = self.toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        stats.register("min", np.min)

        algorithms.eaSimple(
            pop,
            self.toolbox,
            cxpb=0.8,
            mutpb=0.2,
            ngen=generations,
            stats=stats,
            halloffame=hof,
            verbose=True,
        )

        if not hof or hof[0].fitness.values[0] == 0:
            print("\n" + "=" * 50)
            print("COULD NOT FIND A VALID TIMETABLE")
            print(
                "This likely means there are unavoidable time clashes between the required sections of your chosen courses."
            )
            print("Please try a different combination of courses.")
            print("=" * 50)
            return None

        # Build the best timetable. This is now guaranteed to be valid.
        best_ind = hof[0]
        best_timetable = Timetable()
        for i, choice_index in enumerate(best_ind):
            activity, course, sections = self.gene_map[i]
            chosen_section = sections[choice_index]
            best_timetable.add_section(chosen_section)

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


def print_missing_courses(
    timetable: Timetable, selected_courses: List[str], section_groups: Dict
):
    """Print any courses that couldn't be scheduled"""
    scheduled_courses = {sc.class_obj.course for sc in timetable.scheduled_classes}
    missing = set(selected_courses) - scheduled_courses
    if missing:
        print("\n=== Warning: Could Not Schedule ===")
        for course in missing:
            print(f"  - {course}")
            # Show available sections for missing courses
            if course in section_groups:
                print("    Available sections:")
                for section_key in section_groups[course]:
                    print(f"    - {section_key}")


def main():
    print("=== University Timetable Generator ===")
    classes = load_classes_from_csv("classes.csv")
    if not classes:
        print("Could not load any classes from classes.csv. Exiting.")
        return

    print(f"Loaded {len(classes)} classes from CSV")

    user_prefs = get_user_preferences(classes)

    print("\nGenerating timetable based on your preferences...")
    try:
        generator = TimetableGenerator(classes, user_prefs)
        best_timetable = generator.run(generations=150, pop_size=500)

        # ### CHANGED ###: Handle the case where no timetable is returned
        if best_timetable:
            print_timetable(best_timetable)
            print_section_summary(best_timetable)

            print("\n=== Schedule Statistics ===")
            days_used = best_timetable.get_utilized_days()
            print(f"Days used: {days_used} of {len(DAYS)}")
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

            # Calculate average gap between classes
            total_gap = timedelta()
            gap_count = 0
            for day in DAYS:
                day_classes = sorted(
                    best_timetable.schedule[day], key=lambda x: x.start_time
                )
                for i in range(1, len(day_classes)):
                    prev_end = datetime.combine(
                        datetime.today(), day_classes[i - 1].end_time
                    )
                    curr_start = datetime.combine(
                        datetime.today(), day_classes[i].start_time
                    )
                    gap = curr_start - prev_end
                    if gap > timedelta(0):  # Only count positive gaps
                        total_gap += gap
                        gap_count += 1
            avg_gap = total_gap / gap_count if gap_count > 0 else timedelta(0)
            print(f"Average gap between classes: {avg_gap}")

            # Check for consecutive classes
            consecutive_counts = []
            for day in DAYS:
                day_classes = sorted(
                    best_timetable.schedule[day], key=lambda x: x.start_time
                )
                current_streak = 1
                for i in range(1, len(day_classes)):
                    prev_end = datetime.combine(
                        datetime.today(), day_classes[i - 1].end_time
                    )
                    curr_start = datetime.combine(
                        datetime.today(), day_classes[i].start_time
                    )
                    if curr_start - prev_end <= timedelta(
                        minutes=15
                    ):  # Considered consecutive if gap <= 15 mins
                        current_streak += 1
                    else:
                        if current_streak > 1:
                            consecutive_counts.append(current_streak)
                        current_streak = 1
                if current_streak > 1:
                    consecutive_counts.append(current_streak)

            if consecutive_counts:
                print(f"Consecutive classes: {', '.join(map(str, consecutive_counts))}")
            else:
                print("No consecutive classes (more than 1 in a row)")

    except ValueError as e:
        print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()
