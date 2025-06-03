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


@dataclass
class Class:
    id: str
    subject: str
    class_type: str  # "Lecture" or "Tutorial"
    lecturer: str
    duration: int  # minutes
    possible_times: List[Dict[str, List[time]]]


# Predefined days and time slots
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
START_TIME = time(8, 0)  # 8 AM
END_TIME = time(17, 0)  # 5 PM


# Generate all possible time slots
def generate_time_slots():
    slots = []
    current = datetime.combine(datetime.today(), START_TIME)
    end = datetime.combine(datetime.today(), END_TIME)
    while current <= end:
        slots.append(current.time())
        current += timedelta(minutes=TIME_STEP)
    return slots


TIME_SLOTS = generate_time_slots()

# Predefined classes (unchanged from your original)
CLASSES = [
    # Math Classes
    Class(
        "M101",
        "Math",
        "Lecture",
        "Dr. Smith",
        90,
        [{"Mon": [time(9, 0), time(13, 0)]}, {"Tue": [time(10, 0), time(14, 0)]}],
    ),
    Class(
        "M102",
        "Math",
        "Tutorial",
        "Prof. Johnson",
        60,
        [{"Mon": [time(11, 0), time(15, 0)]}, {"Wed": [time(10, 0), time(14, 0)]}],
    ),
    # Physics Classes
    Class(
        "P101",
        "Physics",
        "Lecture",
        "Dr. Brown",
        120,
        [{"Tue": [time(9, 0), time(13, 0)]}, {"Thu": [time(10, 0), time(14, 0)]}],
    ),
    # Chemistry Classes
    Class(
        "C101",
        "Chemistry",
        "Lecture",
        "Prof. Davis",
        90,
        [{"Wed": [time(9, 0), time(13, 0)]}, {"Fri": [time(10, 0), time(14, 0)]}],
    ),
    Class(
        "C102",
        "Chemistry",
        "Tutorial",
        "Dr. Wilson",
        60,
        [{"Thu": [time(11, 0), time(15, 0)]}, {"Fri": [time(9, 0), time(13, 0)]}],
    ),
    # Biology Classes
    Class(
        "B101",
        "Biology",
        "Tutorial",
        "Dr. Taylor",
        90,
        [{"Mon": [time(10, 0), time(14, 0)]}, {"Wed": [time(11, 0), time(15, 0)]}],
    ),
    Class(
        "B102",
        "Biology",
        "Tutorial",
        "Dr. Taylor",
        90,
        [{"Tue": [time(9, 0), time(13, 0)]}, {"Thu": [time(10, 0), time(14, 0)]}],
    ),
    # Computer Science Classes
    Class(
        "CS101",
        "Computer Science",
        "Lecture",
        "Prof. Miller",
        90,
        [{"Mon": [time(9, 0), time(13, 0)]}, {"Wed": [time(10, 0), time(14, 0)]}],
    ),
    Class(
        "CS102",
        "Computer Science",
        "Lecture",
        "Dr. Anderson",
        90,
        [{"Tue": [time(11, 0), time(15, 0)]}, {"Thu": [time(9, 0), time(13, 0)]}],
    ),
    Class(
        "CS103",
        "Computer Science",
        "Tutorial",
        "Dr. Anderson",
        60,
        [{"Fri": [time(10, 0), time(14, 0)]}],
    ),
]

# Subject requirements
SUBJECT_REQUIREMENTS = {
    "Math": {"Lecture": 1, "Tutorial": 1},
    "Physics": {"Lecture": 1},
    "Chemistry": {"Lecture": 1, "Tutorial": 1},
    "Biology": {"Tutorial": 2},
    "Computer Science": {"Lecture": 2, "Tutorial": 1},
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

        # Check lecturer availability
        lecturer = scheduled_class.class_obj.lecturer
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
            subject_times[sc.class_obj.subject].append(
                (sc.day, datetime.combine(datetime.today(), sc.start_time))
            )

        gaps = defaultdict(list)
        for subject, times in subject_times.items():
            times.sort(key=lambda x: x[1])
            for i in range(1, len(times)):
                if times[i][0] == times[i - 1][0]:
                    gap = (times[i][1] - times[i - 1][1]).total_seconds() / 60
                    gaps[subject].append(gap)
        return gaps


class TimetableGenerator:
    def __init__(self, user_preferences):
        self.user_preferences = user_preferences
        self.valid_classes = [
            i
            for i, c in enumerate(CLASSES)
            if c.subject in user_preferences["subjects"]
        ]
        self.setup_deap()

    def setup_deap(self):
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_bool", lambda: random.choice(self.valid_classes))
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self.toolbox.attr_bool,
            n=len(CLASSES),
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register(
            "mutate",
            tools.mutUniformInt,
            low=min(self.valid_classes),
            up=max(self.valid_classes),
            indpb=0.1,
        )
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def evaluate(self, individual: Tuple) -> Tuple[float]:
        timetable = Timetable()
        selected_classes = [
            CLASSES[i] for i in individual if i in self.valid_classes
        ]  # Only preferred subjects
        score = 0

        # Try to schedule each selected class
        for class_obj in selected_classes:
            scheduled = False
            for time_option in class_obj.possible_times:
                for day, start_times in time_option.items():
                    if day not in self.user_preferences["preferred_days"]:
                        continue
                    for start_time in start_times:
                        # Check if time matches user preferences
                        preferred_start = self.user_preferences["preferred_start"]
                        preferred_end = self.user_preferences["preferred_end"]
                        if not (preferred_start <= start_time <= preferred_end):
                            continue

                        end_time = (
                            datetime.combine(datetime.today(), start_time)
                            + timedelta(minutes=class_obj.duration)
                        ).time()
                        sc = ScheduledClass(class_obj, day, start_time, end_time)
                        if timetable.add_class(sc):
                            scheduled = True
                            score += 10  # Base score for scheduling
                            # Bonus for preferred time
                            if preferred_start <= start_time <= preferred_end:
                                score += 5
                            break
                    if scheduled:
                        break
                if scheduled:
                    break

            if not scheduled:
                score -= 5  # Penalty for not scheduling

        # Check subject requirements for preferred subjects only
        subject_counts = defaultdict(lambda: defaultdict(int))
        for sc in timetable.scheduled_classes:
            subject_counts[sc.class_obj.subject][sc.class_obj.class_type] += 1

        for subject in self.user_preferences["subjects"]:
            if subject in SUBJECT_REQUIREMENTS:
                for class_type, count in SUBJECT_REQUIREMENTS[subject].items():
                    if subject_counts[subject][class_type] >= count:
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

        # Build timetable from best individual
        best_ind = hof[0]
        timetable = Timetable()
        for class_idx in best_ind:
            if class_idx not in self.valid_classes:
                continue
            class_obj = CLASSES[class_idx]
            scheduled = False
            for time_option in class_obj.possible_times:
                for day, start_times in time_option.items():
                    for start_time in start_times:
                        end_time = (
                            datetime.combine(datetime.today(), start_time)
                            + timedelta(minutes=class_obj.duration)
                        ).time()
                        sc = ScheduledClass(class_obj, day, start_time, end_time)
                        if timetable.add_class(sc):
                            scheduled = True
                            break
                    if scheduled:
                        break
                if scheduled:
                    break
        return timetable


def get_user_preferences():
    print("Available subjects:", list(SUBJECT_REQUIREMENTS.keys()))
    while True:
        subjects = (
            input("Enter preferred subjects (comma separated): ").strip().split(",")
        )
        subjects = [s.strip() for s in subjects]
        # Validate subjects
        invalid = [s for s in subjects if s not in SUBJECT_REQUIREMENTS]
        if not invalid:
            break
        print(
            f"Invalid subjects: {invalid}. Please choose from {list(SUBJECT_REQUIREMENTS.keys())}"
        )

    print("\nAvailable days:", DAYS)
    preferred_days = (
        input("Enter preferred days (comma separated): ").strip().split(",")
    )
    preferred_days = [d.strip() for d in preferred_days]

    print("\nPreferred time range (HH:MM format)")
    start = input("Earliest preferred start time (e.g., 09:00): ").strip()
    end = input("Latest preferred end time (e.g., 16:00): ").strip()

    return {
        "subjects": subjects,
        "preferred_days": preferred_days,
        "preferred_start": time(*map(int, start.split(":"))),
        "preferred_end": time(*map(int, end.split(":"))),
    }


def print_timetable(timetable: Timetable):
    print("\n=== Optimized Timetable ===")
    for day in DAYS:
        print(f"\n{day}:")
        if not timetable.schedule[day]:
            print("No classes")
            continue
        for sc in sorted(timetable.schedule[day], key=lambda x: x.start_time):
            print(
                f"{sc.start_time.strftime('%H:%M')}-{sc.end_time.strftime('%H:%M')}: "
                f"{sc.class_obj.subject} {sc.class_obj.class_type} with {sc.class_obj.lecturer}"
            )


def main():
    print("=== University Timetable Generator ===")
    user_prefs = get_user_preferences()

    print("\nGenerating timetable based on your preferences...")
    generator = TimetableGenerator(user_prefs)
    best_timetable = generator.run()

    print_timetable(best_timetable)

    # Statistics
    print("\n=== Schedule Statistics ===")
    print(f"Days used: {best_timetable.get_utilized_days()} of {len(DAYS)}")
    print(
        f"Preferred days used: {len([d for d in user_prefs['preferred_days'] if best_timetable.schedule[d]])}"
    )


if __name__ == "__main__":
    main()
