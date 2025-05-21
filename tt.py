from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
import random
import numpy as np
from collections import defaultdict
from deap import base, creator, tools, algorithms
from datetime import time, datetime, timedelta

# Initialize DEAP types
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Constants
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
START_HOUR = 8  # 8 AM
END_HOUR = 18  # 6 PM
TIME_STEP = 30  # 30-minute intervals
MIN_CLASSES = 1  # Minimum number of classes per individual


@dataclass
class ClassType:
    name: str
    duration: int  # in minutes
    min_start: int = 8  # 8 AM
    max_end: int = 18  # 6 PM


@dataclass
class Subject:
    name: str
    class_types: Dict[str, ClassType]
    lecturers: List[str]
    required_count: Dict[str, int]


@dataclass
class ScheduledClass:
    subject: str
    class_type: str
    lecturer: str
    start_time: time
    end_time: time

    def __hash__(self):
        return hash(
            (
                self.subject,
                self.class_type,
                self.lecturer,
                self.start_time.hour,
                self.start_time.minute,
                self.end_time.hour,
                self.end_time.minute,
            )
        )


@dataclass
class Timetable:
    schedule: Dict[str, List[Optional[ScheduledClass]]] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize empty schedule with 30-minute intervals"""
        self.time_slots = self._generate_time_slots()
        for day in DAYS:
            self.schedule[day] = [None] * len(self.time_slots)

    def _generate_time_slots(self):
        """Generate all possible time slots (30-minute intervals)"""
        slots = []
        current = datetime(2000, 1, 1, START_HOUR)  # Dummy date
        end = datetime(2000, 1, 1, END_HOUR)
        while current <= end:
            slots.append(current.time())
            current += timedelta(minutes=TIME_STEP)
        return slots

    def add_class(
        self, day: str, start_time: time, duration: int, class_obj: ScheduledClass
    ) -> bool:
        """Add a class if the time slot is available"""
        try:
            start_idx = self.time_slots.index(start_time)
        except ValueError:
            return False  # Start time not in our slots

        duration_slots = duration // TIME_STEP
        end_idx = start_idx + duration_slots

        if end_idx > len(self.time_slots):
            return False  # Would go past end time

        for i in range(start_idx, end_idx):
            if self.schedule[day][i] is not None:
                return False  # Overlapping class

        for i in range(start_idx, end_idx):
            self.schedule[day][i] = class_obj
        return True

    def get_lecturer_schedule(
        self, lecturer: str
    ) -> Dict[str, List[Tuple[time, time]]]:
        """Get all time ranges when a lecturer is busy"""
        busy_times = defaultdict(list)
        for day in DAYS:
            current_start = None
            current_end = None
            for i, slot in enumerate(self.time_slots):
                if self.schedule[day][i] and self.schedule[day][i].lecturer == lecturer:
                    if current_start is None:
                        current_start = slot
                    current_end = self.time_slots[min(i + 1, len(self.time_slots) - 1)]
                else:
                    if current_start is not None:
                        busy_times[day].append((current_start, current_end))
                        current_start = None
            if current_start is not None:
                busy_times[day].append((current_start, current_end))
        return busy_times


class TimetableGenerator:
    def __init__(self, subjects: List[Subject], population_size: int = 100):
        self.subjects = subjects
        self.population_size = population_size
        self.setup_deap()

    def setup_deap(self):
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self.init_individual)
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", self.crossover)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def init_individual(self) -> creator.Individual:
        """Create random timetable with variable class durations"""
        while True:  # Keep trying until we get a valid individual
            timetable = Timetable()
            scheduled_count = 0

            for subject in self.subjects:
                for class_type, count in subject.required_count.items():
                    for _ in range(count):
                        if self.schedule_class(
                            timetable, subject, subject.class_types[class_type]
                        ):
                            scheduled_count += 1

            if scheduled_count >= MIN_CLASSES:  # Ensure minimum classes
                encoded = self.encode_timetable(timetable)
                if len(encoded) >= 6:  # At least one complete class
                    return creator.Individual(encoded)

    def schedule_class(
        self, timetable: Timetable, subject: Subject, class_type: ClassType
    ) -> bool:
        """Schedule a single class with proper duration"""
        lecturer = random.choice(subject.lecturers)
        attempts = 0

        while attempts < 100:
            day = random.choice(DAYS)

            # Generate valid start times considering class type constraints
            possible_starts = [
                t
                for t in timetable.time_slots
                if t.hour >= class_type.min_start
                and (
                    datetime.combine(datetime.today(), t)
                    + timedelta(minutes=class_type.duration)
                )
                .time()
                .hour
                <= class_type.max_end
            ]

            if not possible_starts:
                attempts += 1
                continue

            start_time = random.choice(possible_starts)
            class_obj = ScheduledClass(
                subject.name,
                class_type.name,
                lecturer,
                start_time,
                (
                    datetime.combine(datetime.today(), start_time)
                    + timedelta(minutes=class_type.duration)
                ).time(),
            )

            if timetable.add_class(day, start_time, class_type.duration, class_obj):
                return True
            attempts += 1
        return False

    def encode_timetable(self, timetable: Timetable) -> List:
        """Flatten timetable for DEAP"""
        encoded = []
        for day in DAYS:
            for i, slot in enumerate(timetable.time_slots):
                if class_obj := timetable.schedule[day][i]:
                    # Only encode start of each class (skip continuation slots)
                    if i == 0 or timetable.schedule[day][i - 1] != class_obj:
                        encoded.extend(
                            [
                                day,
                                class_obj.start_time.strftime("%H:%M"),
                                class_obj.end_time.strftime("%H:%M"),
                                class_obj.subject,
                                class_obj.class_type,
                                class_obj.lecturer,
                            ]
                        )
        return encoded

    def decode_timetable(self, individual: creator.Individual) -> Timetable:
        """Convert flat list back to Timetable"""
        timetable = Timetable()
        i = 0
        while i < len(individual):
            day = individual[i]
            start_time = datetime.strptime(individual[i + 1], "%H:%M").time()
            end_time = datetime.strptime(individual[i + 2], "%H:%M").time()
            subject = individual[i + 3]
            class_type = individual[i + 4]
            lecturer = individual[i + 5]
            i += 6

            # Find the subject
            subject_obj = next((s for s in self.subjects if s.name == subject), None)
            if not subject_obj:
                continue

            # Verify class type exists for this subject
            if class_type not in subject_obj.class_types:
                continue

            # Calculate duration
            duration = int(
                (
                    datetime.combine(datetime.today(), end_time)
                    - datetime.combine(datetime.today(), start_time)
                ).total_seconds()
                / 60
            )

            class_obj = ScheduledClass(
                subject, class_type, lecturer, start_time, end_time
            )
            timetable.add_class(day, start_time, duration, class_obj)
        return timetable

    def evaluate(self, individual: creator.Individual) -> Tuple[float]:
        """Enhanced fitness function with duration awareness"""
        timetable = self.decode_timetable(individual)
        score = 0
        scheduled_counts = defaultdict(lambda: defaultdict(int))
        lecturer_workload = defaultdict(int)

        # Track scheduled classes
        for day in DAYS:
            current_classes = set()
            for i, slot in enumerate(timetable.time_slots):
                if class_obj := timetable.schedule[day][i]:
                    if class_obj not in current_classes:
                        current_classes.add(class_obj)
                        scheduled_counts[class_obj.subject][class_obj.class_type] += 1
                        duration = (
                            datetime.combine(datetime.today(), class_obj.end_time)
                            - datetime.combine(datetime.today(), class_obj.start_time)
                        ).seconds // 60
                        lecturer_workload[class_obj.lecturer] += duration

        # Hard constraints
        for subject in self.subjects:
            for class_type, required in subject.required_count.items():
                missing = max(0, required - scheduled_counts[subject.name][class_type])
                score -= 1000 * missing

        # Soft constraints
        if lecturer_workload:
            avg_workload = sum(lecturer_workload.values()) / len(lecturer_workload)
            for workload in lecturer_workload.values():
                score -= 0.1 * abs(workload - avg_workload)

        return (max(score, 0.1),)

    def crossover(
        self, ind1: creator.Individual, ind2: creator.Individual
    ) -> Tuple[creator.Individual, creator.Individual]:
        """Custom crossover that ensures valid individuals"""
        if len(ind1) < 6 or len(ind2) < 6:
            return ind1, ind2  # Not enough material to crossover

        # Ensure crossover points are valid
        size = min(len(ind1), len(ind2))
        cxpoint1 = random.randint(1, size // 6) * 6  # Align with class boundaries
        cxpoint2 = random.randint(1, size // 6) * 6

        if cxpoint1 > cxpoint2:
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = (
            ind2[cxpoint1:cxpoint2],
            ind1[cxpoint1:cxpoint2],
        )

        return ind1, ind2

    def mutate(self, individual: creator.Individual) -> Tuple[creator.Individual]:
        """Mutation that respects class durations"""
        mutant = list(individual)
        if random.random() < 0.2 and len(mutant) >= 6:
            # Select random class to move
            class_idx = random.randrange(0, len(mutant) // 6) * 6
            class_info = mutant[class_idx : class_idx + 6]

            # Remove from current position
            del mutant[class_idx : class_idx + 6]

            # Try to reschedule
            temp_timetable = self.decode_timetable(creator.Individual(mutant))
            subject = next((s for s in self.subjects if s.name == class_info[3]), None)
            if subject and class_info[4] in subject.class_types:
                class_type = subject.class_types[class_info[4]]
                self.schedule_class(temp_timetable, subject, class_type)

            # Update individual
            mutant = self.encode_timetable(temp_timetable)

        return (creator.Individual(mutant),)

    def run(self, generations: int = 100) -> Timetable:
        """Run the evolutionary algorithm"""
        pop = self.toolbox.population(n=self.population_size)
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

        best = self.decode_timetable(hof[0])

        # Verification
        print("\nVerification:")
        for subject in self.subjects:
            for class_type, count in subject.required_count.items():
                scheduled = sum(
                    1
                    for day in DAYS
                    for slot in best.time_slots
                    if best.schedule[day][best.time_slots.index(slot)]
                    and best.schedule[day][best.time_slots.index(slot)].subject
                    == subject.name
                    and best.schedule[day][best.time_slots.index(slot)].class_type
                    == class_type
                )
                print(f"{subject.name} {class_type}: {scheduled}/{count}")

        return best


if __name__ == "__main__":
    # Example configuration with varying durations
    subjects = [
        Subject(
            "Math",
            {
                "Lecture": ClassType("Lecture", 90, 9, 16),
                "Tutorial": ClassType("Tutorial", 60, 10, 17),
            },
            ["Dr. Smith", "Prof. Johnson"],
            {"Lecture": 1, "Tutorial": 1},
        ),
        Subject(
            "Physics",
            {"Lecture": ClassType("Lecture", 120, 8, 15)},
            ["Dr. Brown"],
            {"Lecture": 1},
        ),
        Subject(
            "Biology",
            {"Tutorial": ClassType("Tutorial", 90, 9, 16)},
            ["Dr. Taylor"],
            {"Tutorial": 2},
        ),
    ]

    generator = TimetableGenerator(subjects, population_size=100)
    best_timetable = generator.run(generations=100)

    # Print the timetable
    print("\nOptimized Timetable with Variable Durations:")
    for day in DAYS:
        print(f"\n{day}:")
        current_classes = set()
        for slot in best_timetable.time_slots:
            class_obj = best_timetable.schedule[day][
                best_timetable.time_slots.index(slot)
            ]
            if class_obj and class_obj not in current_classes:
                current_classes.add(class_obj)
                print(
                    f"{class_obj.start_time.strftime('%H:%M')}-{class_obj.end_time.strftime('%H:%M')}: "
                    f"{class_obj.subject} ({class_obj.class_type}) with {class_obj.lecturer}"
                )
