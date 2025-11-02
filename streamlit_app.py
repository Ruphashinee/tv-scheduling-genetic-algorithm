import streamlit as st
import pandas as pd
import random
import csv

# --- PART A: THE GENETIC ALGORITHM "ENGINE" ---
# This is all the backend logic for the GA

def read_csv_to_dict(file_path):
    """Reads the CSV file and returns a dictionary of ratings."""
    program_ratings = {}
    try:
        with open(file_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            try:
                header = next(reader) # Skip header
            except StopIteration:
                st.error(f"Error: The file '{file_path}' is empty.")
                return {}
            
            for row in reader:
                if len(row) > 1:
                    program = row[0]
                    try:
                        ratings = [float(x) for x in row[1:]]
                        program_ratings[program] = ratings
                    except ValueError:
                        st.warning(f"Skipping row for '{program}': non-numeric rating.")
    
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found.")
        st.info("Using sample data as a fallback.")
        # Fallback sample data
        program_ratings = {
            'News': [4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5],
            'Movie': [3.0, 3.5, 4.0, 4.5, 5.0, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            'Sports': [4.0, 4.5, 5.0, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.0, 4.5]
        }
    return program_ratings

def fitness_function(schedule, ratings_data, schedule_length):
    """Calculates the total fitness of a given schedule."""
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        if program in ratings_data:
            if time_slot < len(ratings_data[program]):
                total_rating += ratings_data[program][time_slot]
    return total_rating

def create_random_schedule(all_programs, schedule_length):
    """Creates a single, completely random schedule."""
    # This uses random.choice (allows repeats), which is correct.
    return [random.choice(all_programs) for _ in range(schedule_length)]

def crossover(schedule1, schedule2, schedule_length):
    """Performs single-point crossover (correct for assignment)."""
    if len(schedule1) < 2 or len(schedule2) < 2:
        return schedule1, schedule2
    crossover_point = random.randint(1, schedule_length - 1)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2

def mutate(schedule, all_programs, schedule_length):
    """Mutates a schedule by changing one random gene (correct for assignment)."""
    schedule_copy = schedule.copy()
    mutation_point = random.randint(0, schedule_length - 1)
    new_program = random.choice(all_programs)
    schedule_copy[mutation_point] = new_program
    return schedule_copy

def genetic_algorithm(ratings_data, all_programs, schedule_length,
                      generations=100, population_size=50,
                      crossover_rate=0.8, mutation_rate=0.2, elitism_size=2):
    """Runs the genetic algorithm."""
    
    # Initialize population
    population = [create_random_schedule(all_programs, schedule_length) for _ in range(population_size)]
    
    best_schedule_ever = []
    best_fitness_ever = 0

    for generation in range(generations):
.
        pop_with_fitness = []
        for schedule in population:
            fitness = fitness_function(schedule, ratings_data, schedule_length)
            pop_with_fitness.append((schedule, fitness))
            
            if fitness > best_fitness_ever:
                best_fitness_ever = fitness
                best_schedule_ever = schedule

        pop_with_fitness.sort(key=lambda x: x[1], reverse=True)
        
        new_population = []
        # Elitism
        for i in range(elitism_size):
            new_population.append(pop_with_fitness[i][0])

        # Fill the rest of the new population
        while len(new_population) < population_size:
            parent1 = random.choice(pop_with_fitness[:population_size // 2])[0]
            parent2 = random.choice(pop_with_fitness[:population_size // 2])[0]

            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2, schedule_length)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < mutation_rate:
                child1 = mutate(child1, all_programs, schedule_length)
            if random.random() < mutation_rate:
                child2 = mutate(child2, all_programs, schedule_length)

            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population

    return best_schedule_ever, best_fitness_ever

# --- PART B: THE STREAMLIT WEB APPLICATION ---
# This code builds the user interface

st.title("📺 Genetic Algorithm - TV Program Scheduling Optimizer")

# --- 1. Load Data ---
file_path = 'program_ratings_modified.csv' 
ratings = read_csv_to_dict(file_path)

# *** NEW LINES ADDED HERE ***
st.subheader("📊 Program Ratings Dataset")
try:
    # We read the file *again* with Pandas just to display it easily
    df_display = pd.read_csv(file_path)
    st.dataframe(df_display)
except FileNotFoundError:
    st.error(f"Could not find {file_path} to display.")
# *** END OF NEW LINES ***


if ratings:
    all_programs = list(ratings.keys())
    all_time_slots = list(range(6, 24)) # 6:00 to 23:00
    SCHEDULE_LENGTH = len(all_time_slots) # 18 slots
    
    st.write(f"Successfully loaded {len(all_programs)} programs for optimization.")
    st.write(f"Schedule will be optimized for {SCHEDULE_LENGTH} time slots (6:00 to 23:00).")

    # --- 2. Sidebar for Parameter Input ---
    # This fulfills Instruction 3
    st.sidebar.header("🧬 Set GA Parameters")

    # Parameters for Trial 1
    st.sidebar.subheader("Trial 1")
    co_r_1 = st.sidebar.slider("Crossover Rate (Trial 1)", 0.0, 0.95, 0.8, 0.05)
    mut_r_1 = st.sidebar.slider("Mutation Rate (Trial 1)", 0.01, 0.05, 0.02, 0.01)

    # Parameters for Trial 2
    st.sidebar.subheader("Trial 2")
    co_r_2 = st.sidebar.slider("Crossover Rate (Trial 2)", 0.0, 0.95, 0.9, 0.05)
    mut_r_2 = st.sidebar.slider("Mutation Rate (Trial 2)", 0.01, 0.05, 0.01, 0.01)

    # Parameters for Trial 3
    st.sidebar.subheader("Trial 3")
    co_r_3 = st.sidebar.slider("Crossover Rate (Trial 3)", 0.0, 0.95, 0.7, 0.05)
    mut_r_3 = st.sidebar.slider("Mutation Rate (Trial 3)", 0.01, 0.05, 0.05, 0.01)

    # --- 3. Run Button and Display Results ---
    # This fulfills Instructions 4 and 5
    if st.sidebar.button("🚀 Run All 3 Trials"):
        
        # --- Run Trial 1 ---
        st.header("Trial 1 Results")
        st.write(f"**Parameters:** Crossover Rate = `{co_r_1}`, Mutation Rate = `{mut_r_1}`")
        schedule_1, fitness_1 = genetic_algorithm(
            ratings_data=ratings, all_programs=all_programs, schedule_length=SCHEDULE_LENGTH,
            crossover_rate=co_r_1, mutation_rate=mut_r_1
        )
        # Create DataFrame for table display
        df_1 = pd.DataFrame({
            "Time Slot": [f"{h:02d}:00" for h in all_time_slots],
            "Scheduled Program": schedule_1
        })
        st.dataframe(df_1)
        st.write(f"**Best Fitness Score:** `{fitness_1:.2f}`")
        st.markdown("---")

        # --- Run Trial 2 ---
        st.header("Trial 2 Results")
        st.write(f"**Parameters:** Crossover Rate = `{co_r_2}`, Mutation Rate = `{mut_r_2}`")
        schedule_2, fitness_2 = genetic_algorithm(
            ratings_data=ratings, all_programs=all_programs, schedule_length=SCHEDULE_LENGTH,
            crossover_rate=co_r_2, mutation_rate=mut_r_2
        )
        df_2 = pd.DataFrame({
            "Time Slot": [f"{h:02d}:00" for h in all_time_slots],
            "Scheduled Program": schedule_2
        })
        st.dataframe(df_2)
        st.write(f"**Best Fitness Score:** `{fitness_2:.2f}`")
        st.markdown("---")

        # --- Run Trial 3 ---
        st.header("Trial 3 Results")
        st.write(f"**Parameters:** Crossover Rate = `{co_r_3}`, Mutation Rate = `{mut_r_3}`")
        schedule_3, fitness_3 = genetic_algorithm(
            ratings_data=ratings, all_programs=all_programs, schedule_length=SCHEDULE_LENGTH,
            crossover_rate=co_r_3, mutation_rate=mut_r_3
        )
        df_3 = pd.DataFrame({
            "Time Slot": [f"{h:02d}:00" for h in all_time_slots],
            "Scheduled Program": schedule_3
        })
        st.dataframe(df_3)
        st.write(f"**Best Fitness Score:** `{fitness_3:.2f}`")
        st.markdown("---")

else:
    st.error("Could not load any program data. Please check the file path and CSV content.")
