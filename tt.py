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
    def __init__(self, section_groups=None):
        self.schedule = {day: [] for day in DAYS}
        self.scheduled_classes = []
        self.scheduled_sections = defaultdict(set)  # course: set of section keys
        self.section_groups = section_groups if section_groups else {}

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
        self.setup_deap()

    def setup_deap(self):
        self.toolbox = base.Toolbox()

        # Create a list of all possible section choices
        self.section_choices = []
        self.section_info = []  # Stores (course, activity, section_classes)

        for course in self.user_preferences["courses"]:
            if course in self.section_groups:
                for section_key, section_classes in self.section_groups[course].items():
                    activity = section_classes[0].activity
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
        timetable = Timetable(self.section_groups)
        score = 0

        # Track which sections we've selected for each course and activity type
        selected_sections = defaultdict(lambda: defaultdict(list))
        for idx, selected in enumerate(individual):
            if selected:
                course, activity, section_classes = self.section_info[idx]
                selected_sections[course][activity].append(section_classes)

        # Validate and score each selected section
        for course in self.user_preferences["courses"]:
            # Check we have exactly one lecture and one tutorial (if available)
            lectures = selected_sections[course]["Lecture"]
            tutorials = selected_sections[course]["Tutorial"]

            # Check if course has lecture sections available
            has_lecture_available = any(
                s[1] == "Lecture" for s in self.section_info if s[0] == course
            )
            # Check if course has tutorial sections available
            has_tutorial_available = any(
                s[1] == "Tutorial" for s in self.section_info if s[0] == course
            )

            # Penalize if not exactly one lecture (when available)
            if has_lecture_available:
                if len(lectures) != 1:
                    score -= 100000  # Extremely heavy penalty for wrong number of lecture sections
            # Penalize if not exactly one tutorial (when available)
            if has_tutorial_available:
                if len(tutorials) != 1:
                    score -= 100000  # Extremely heavy penalty for wrong number of tutorial sections

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
                        score += 1000 * len(section_classes)  # High base score
                        if matches_prefs:
                            score += 500 * len(section_classes)  # High preference bonus
                    else:
                        score -= 500 * len(section_classes)  # High penalty for conflict

        # Additional scoring based on timetable quality
        if timetable.meets_requirements(self.user_preferences["courses"]):
            # Reward for fewer days used
            days_used = timetable.get_utilized_days()
            score += (len(DAYS) - days_used) * 1500  # More reward for fewer days

            # Reward for consecutive days
            score += timetable.get_consecutive_days_score() * 1000

            # Reward for good gaps between classes
            gap_score = sum(
                timetable.get_day_gaps_score(day)
                for day in DAYS
                if timetable.schedule[day]
            )
            if days_used > 0:
                gap_score /= days_used
            score += gap_score * 500

            # Reward for preferred days
            preferred_days_count = len(
                [
                    d
                    for d in self.user_preferences["preferred_days"]
                    if any(sc.day == d for sc in timetable.scheduled_classes)
                ]
            )
            score += preferred_days_count * 300

            # Reward for earlier start times (to help consolidate into fewer days)
            avg_start_time = (
                sum(
                    datetime.combine(datetime.today(), sc.start_time).timestamp()
                    for sc in timetable.scheduled_classes
                )
                / len(timetable.scheduled_classes)
                if timetable.scheduled_classes
                else 0
            )
            score += (86400 - avg_start_time) / 3600 * 100  # Reward earlier start times
        else:
            # Extremely heavy penalty for missing requirements
            score -= 1000000

        return (max(score, 1),)

    def run(self, generations=500) -> Timetable:
        pop = self.toolbox.population(n=1000)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)

        # Run the genetic algorithm
        algorithms.eaSimple(
            pop,
            self.toolbox,
            cxpb=0.8,
            mutpb=0.1,
            ngen=generations,
            stats=stats,
            halloffame=hof,
            verbose=True,
        )

        # Build the best timetable
        best_ind = hof[0]
        best_timetable = Timetable(self.section_groups)

        # First add required sections (one lecture and one tutorial per course)
        for course in self.user_preferences["courses"]:
            # Find selected lecture sections
            lecture_sections = []
            for idx, (c, a, sc) in enumerate(self.section_info):
                if c == course and a == "Lecture" and best_ind[idx]:
                    lecture_sections.append(sc)

            # Add exactly one lecture section if available
            if lecture_sections:
                # Sort by preference match and earlier times
                lecture_sections.sort(
                    key=lambda sc: (
                        all(
                            cls.days in self.user_preferences["preferred_days"]
                            and self.user_preferences["preferred_start"]
                            <= cls.start_time
                            <= self.user_preferences["preferred_end"]
                            for cls in sc
                        ),
                        sc[0].start_time,  # Earlier times first
                    ),
                    reverse=True,
                )

                # Try all possible lecture sections until one fits
                for section in lecture_sections:
                    if best_timetable.add_section(section):
                        break
                else:
                    print(
                        f"\nWarning: Could not schedule any lecture section for {course}"
                    )

            # Find selected tutorial sections
            tutorial_sections = []
            for idx, (c, a, sc) in enumerate(self.section_info):
                if c == course and a == "Tutorial" and best_ind[idx]:
                    tutorial_sections.append(sc)

            # Add exactly one tutorial section if available
            if tutorial_sections:
                # Sort by preference match and earlier times
                tutorial_sections.sort(
                    key=lambda sc: (
                        all(
                            cls.days in self.user_preferences["preferred_days"]
                            and self.user_preferences["preferred_start"]
                            <= cls.start_time
                            <= self.user_preferences["preferred_end"]
                            for cls in sc
                        ),
                        sc[0].start_time,  # Earlier times first
                    ),
                    reverse=True,
                )

                # Try all possible tutorial sections until one fits
                for section in tutorial_sections:
                    if best_timetable.add_section(section):
                        break
                else:
                    print(
                        f"\nWarning: Could not schedule any tutorial section for {course}"
                    )

        # Final check to ensure all requirements are met
        if not best_timetable.meets_requirements(self.user_preferences["courses"]):
            print(
                "\nWarning: Could not find a valid schedule meeting all requirements!"
            )
            print("This may be due to conflicting class times or unavailable sections.")
            print("Please try adjusting your preferences or course selection.")

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

    # Load classes from CSV
    classes = load_classes_from_csv("classes.csv")
    print(f"Loaded {len(classes)} classes from CSV")
    section_groups = group_classes_by_section(classes)

    # Get user preferences
    user_prefs = get_user_preferences(classes)

    print("\nGenerating timetable based on your preferences...")
    generator = TimetableGenerator(classes, user_prefs)
    best_timetable = generator.run(generations=500)

    print_timetable(best_timetable)
    print_section_summary(best_timetable)
    print_missing_courses(best_timetable, user_prefs["courses"], section_groups)

    # Statistics
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
        day_classes = sorted(best_timetable.schedule[day], key=lambda x: x.start_time)
        for i in range(1, len(day_classes)):
            prev_end = datetime.combine(datetime.today(), day_classes[i - 1].end_time)
            curr_start = datetime.combine(datetime.today(), day_classes[i].start_time)
            gap = curr_start - prev_end
            if gap > timedelta(0):  # Only count positive gaps
                total_gap += gap
                gap_count += 1
    avg_gap = total_gap / gap_count if gap_count > 0 else timedelta(0)
    print(f"Average gap between classes: {avg_gap}")

    # Check for consecutive classes
    consecutive_counts = []
    for day in DAYS:
        day_classes = sorted(best_timetable.schedule[day], key=lambda x: x.start_time)
        current_streak = 1
        for i in range(1, len(day_classes)):
            prev_end = datetime.combine(datetime.today(), day_classes[i - 1].end_time)
            curr_start = datetime.combine(datetime.today(), day_classes[i].start_time)
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


if __name__ == "__main__":
    main()
