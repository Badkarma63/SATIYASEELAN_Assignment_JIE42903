import streamlit as st
import pandas as pd
import random
# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="GA Scheduling", layout="wide")

st.title("📺 Genetic Algorithm for Program Scheduling")
st.write(
    "This application uses a Genetic Algorithm to optimise TV program scheduling "
    "by maximising total audience ratings across time slots."
)

# ===================== LOAD DATA =====================
@st.cache_data
def load_data():
    df = pd.read_csv("program_ratings.csv")
    return df

df = load_data()

st.subheader("📊 Program Ratings Dataset")
st.dataframe(df)

# ===================== PREPARE DATA =====================
programs = df["Type of Program"].tolist()
time_slots = df.columns[1:].tolist()

ratings = {}
for _, row in df.iterrows():
    ratings[row["Type of Program"]] = row[1:].tolist()

# ===================== PARAMETERS =====================
st.sidebar.header("⚙ GA Parameters")

GEN = st.sidebar.slider("Generations", 10, 300, 100)
POP = st.sidebar.slider("Population Size", 10, 100, 50)
EL_S = st.sidebar.slider("Elitism Size", 1, 5, 2)

# ---------------- TRIAL 1 ----------------
st.sidebar.markdown("---")
st.sidebar.subheader("🔁 Trial 1")
CO_R1 = st.sidebar.slider("Crossover Rate (T1)", 0.1, 1.0, 0.8, key="co1")
MUT_R1 = st.sidebar.slider("Mutation Rate (T1)", 0.01, 0.05, 0.02, key="mut1")
run_t1 = st.sidebar.button("▶ Run Trial 1")

# ---------------- TRIAL 2 ----------------
st.sidebar.markdown("---")
st.sidebar.subheader("🔁 Trial 2")
CO_R2 = st.sidebar.slider("Crossover Rate (T2)", 0.1, 1.0, 0.85, key="co2")
MUT_R2 = st.sidebar.slider("Mutation Rate (T2)", 0.01, 0.05, 0.02, key="mut2")
run_t2 = st.sidebar.button("▶ Run Trial 2")

# ---------------- TRIAL 3 ----------------
st.sidebar.markdown("---")
st.sidebar.subheader("🔁 Trial 3")
CO_R3 = st.sidebar.slider("Crossover Rate (T3)", 0.1, 1.0, 0.95, key="co3")
MUT_R3 = st.sidebar.slider("Mutation Rate (T3)", 0.0, 0.05, 0.03, key="mut3")
run_t3 = st.sidebar.button("▶ Run Trial 3")

# ---------------- RUN ALL ----------------
st.sidebar.markdown("## 🚀 Final Experiment")
run_all = st.sidebar.button("▶ Run ALL Trials")

# ===================== FITNESS FUNCTION =====================
def fitness_function(schedule):
    total = 0
    for i, program in enumerate(schedule):
        total += ratings[program][i]
    return total

# ===================== INITIAL POPULATION =====================
def generate_schedule():
    schedule = programs.copy()
    random.shuffle(schedule)
    return schedule

# ===================== GA OPERATORS =====================
def crossover(p1, p2):
    point = random.randint(1, len(p1) - 2)
    c1 = p1[:point] + p2[point:]
    c2 = p2[:point] + p1[point:]
    return c1, c2

def mutate(schedule):
    # Swap two programs to avoid duplicates
    i, j = random.sample(range(len(schedule)), 2)
    schedule[i], schedule[j] = schedule[j], schedule[i]
    return schedule

# ===================== GENETIC ALGORITHM =====================
def genetic_algorithm(crossover_rate, mutation_rate):
    population = [generate_schedule() for _ in range(POP)]

    for _ in range(GEN):
        population.sort(key=fitness_function, reverse=True)
        new_pop = population[:EL_S]

        while len(new_pop) < POP:
            p1, p2 = random.sample(population, 2)

            if random.random() < crossover_rate:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            if random.random() < mutation_rate:
                c1 = mutate(c1)
            if random.random() < mutation_rate:
                c2 = mutate(c2)

            new_pop.extend([c1, c2])

        population = new_pop[:POP]

    best = max(population, key=fitness_function)
    return best, fitness_function(best)

# ===================== RUN GA FOR ALL TRIALS =====================
def run_and_display(trial_name, co_r, mut_r):
    best_schedule, score = genetic_algorithm(co_r, mut_r)

    # Align schedule length with time slots
    if len(best_schedule) < len(time_slots):
        final_schedule = (best_schedule * ((len(time_slots) // len(best_schedule)) + 1))[:len(time_slots)]
    else:
        final_schedule = best_schedule[:len(time_slots)]

    st.subheader(f"{trial_name}")
    st.success(f"CO_R = {co_r} | MUT_R = {mut_r} | Total Rating = {score}")

    result = pd.DataFrame({
        "Time Slot": time_slots,
        "Program": final_schedule
    })

    st.table(result)

if run_t1:
    run_and_display("🔁 Trial 1 Results", CO_R1, MUT_R1)

if run_t2:
    run_and_display("🔁 Trial 2 Results", CO_R2, MUT_R2)

if run_t3:
    run_and_display("🔁 Trial 3 Results", CO_R3, MUT_R3)

if run_all:
    st.subheader("📊 Results for ALL Trials")
    run_and_display("🔁 Trial 1 Results", CO_R1, MUT_R1)
    run_and_display("🔁 Trial 2 Results", CO_R2, MUT_R2)
    run_and_display("🔁 Trial 3 Results", CO_R3, MUT_R3)
