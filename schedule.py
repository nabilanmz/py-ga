import time
import random
import os


# File parsing utilities
def read_student_file(filename):
    with open(filename, "r") as f:
        data = f.read().strip()
    # Each record separated by whitespace
    records = data.split()
    # Split each record by '$'
    return [rec.split("$") for rec in records]


def read_lecturer_file(filename):
    lecturers = {}
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("&")
            code = parts[0]
            # Filter out empty strings
            names = [p for p in parts[1:] if p]
            lecturers[code] = names
    return lecturers


def read_lecsize_file(filename):
    sizes = {}
    with open(filename, "r") as f:
        for line in f:
            line = line.strip().rstrip("$")
            if not line:
                continue
            parts = line.split("$")
            code, capacity, tut = parts
            sizes[code] = {"capacity": int(capacity), "tut": int(tut)}
    return sizes


def read_timeslot_file(filename):
    slots = []
    total = None
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("TOTAL:"):
                total = int(line.split(":")[1])
            else:
                parts = line.split("\t")
                day = parts[0]
                times = parts[1:]
                for tm in times:
                    slots.append({"day": day, "time": tm})
    return slots, total


def read_lec_file(filename):
    lec = {}
    with open(filename, "r") as f:
        for line in f:
            line = line.strip().rstrip("$")
            if not line:
                continue
            code, lec_count, tut_count = line.split("$")
            lec[code] = {"lec": int(lec_count), "tut": int(tut_count)}
    return lec


def read_penalty_file(filename):
    penalty = {}
    with open(filename, "r") as f:
        for line in f:
            line = line.strip().rstrip("$")
            if not line:
                continue
            key, val = line.split("$")
            penalty[int(key)] = int(val)
    return penalty


# Placeholder implementations for GA subroutines
def generate_course_code(student_records):
    # Implement mapping courses to indices
    pass


def init_constructors(lec_file_content):
    # Initialize course structures
    pass


def create_clash_tables():
    # Build student and lecturer clash tables
    pass


def fitness(chromosome):
    # Compute fitness value for a chromosome
    return random.random()


def select_parents(population, fitnesses):
    # Select parents based on fitness
    return population[0], population[1]


def crossover(parent1, parent2):
    # Uniform crossover
    child1, child2 = parent1[:], parent2[:]
    return [child1, child2]


def mutate(chromosome, rate):
    # Random mutation
    return chromosome


def repair(chromosome):
    # Correct clashes
    return chromosome


# Main schedule function
def schedule():
    # Start timer
    start = time.time()

    # Configuration
    student_file = "student.dat"
    lecturer_file = "lecturer.dat"
    lec_size_file = "lecsize.dat"
    timeslot_file = "timeslot"
    lec_file = "lec.dat"
    crs_file = "crs"
    tv_file = "tv"
    staff_file = "staff"
    course_file = "course"
    gensize = 10
    popsize = 10
    choice = 1
    crossover_rate = 0.6
    mutation_rate = 0.08
    vdm_rate = 0.2

    # Read input files
    students = read_student_file(student_file)
    lecturers = read_lecturer_file(lecturer_file)
    lec_sizes = read_lecsize_file(lec_size_file)
    slots, total_slots = read_timeslot_file(timeslot_file)
    courses_info = read_lec_file(lec_file)
    crs_penalty = read_penalty_file(crs_file)
    tv_penalty = read_penalty_file(tv_file)
    staff_penalty = read_penalty_file(staff_file)
    course_penalty = read_penalty_file(course_file)

    # TODO: Generate course codes, initialize structures
    generate_course_code(students)
    init_constructors(courses_info)
    create_clash_tables()

    # Initialize population: list of chromosomes (lists of slot assignments)
    CHOLEN = len(courses_info)
    population = [[-1] * CHOLEN for _ in range(popsize)]

    # Initialize random seed
    random.seed()

    # GA main loop
    for gen in range(gensize):
        # Evaluate fitness
        fitnesses = [fitness(ch) for ch in population]

        # Create next generation
        new_population = []
        while len(new_population) < popsize:
            # Selection and variation
            if choice == 1:
                p1, p2 = select_parents(population, fitnesses)
                children = crossover(p1, p2)
            else:
                # Other operators (mutation, directed)
                p1 = select_parents(population, fitnesses)[0]
                child = mutate(p1, mutation_rate)
                children = [child]

            # Repair and add children
            for ch in children:
                ch = repair(ch)
                new_population.append(ch)
                if len(new_population) >= popsize:
                    break

        population = new_population
        print(f"Generation {gen+1}/{gensize} complete")

    # Find best solution
    best = max(population, key=fitness)
    best_fit = fitness(best)
    print(f"Best fitness: {best_fit}")

    # Output results
    output_file = "schedule_out.txt"
    with open(output_file, "w") as f:
        f.write(str(best))

    print(f"Schedule saved to {output_file}")
    print(f"Elapsed time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    schedule()
