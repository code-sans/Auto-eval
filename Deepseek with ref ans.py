import subprocess
import re

# Function to run Ollama model locally with DeepSeek
def run_ollama_model(model_name, prompt):
    try:
        result = subprocess.run(
            ['ollama', 'run', model_name],
            input=prompt, text=True, capture_output=True, check=True, encoding="utf-8"  # Ensure UTF-8 encoding
        )
        return result.stdout.strip()  # Return cleaned output
    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama model: {e}")
        return None


# Function to preprocess and evaluate answers
def evaluate_student_answer(reference_answer, student_answer):
    # AI model prompt with step-wise evaluation
    prompt = f"""
    Reference Answer: {reference_answer}
    Student Answer: {student_answer}

    TASK:
    mandatory : do not complete the student answer by adding points , just preprocess it .
    1. **Preprocess** the student's answer by fixing grammar and sentence structure. DO NOT penalize for spelling mistakes. and do not complete the answer in your preprocessing part(strictly ).
    2. **Break the answer into logical steps** matching the reference answer.
    3. **Evaluate step-wise correctness in student asnwer , do not provide marks for irrelavant answer .**:
       - Each correct stage gets 1 mark (out of 4).
       - Provide feedback for each stage (correct, needs improvement, or incorrect).
    4. **Strictly focus on content accuracy.**
    5. If any stage/step of answer is not present assignt 0 marks for that or based on the relvance if present.

    OUTPUT FORMAT:
    - Corrected Answer: (fixed version of student answer)
    - Step-wise Evaluation:
      - Stage 1: (score: a out of 1)
      - Stage 2: (score: b out of 1)
      - Stage 3: (score: c out of 1)
      - Stage 4: (score: d out of 1)

    
    - Final Score: (all stages sum)

    I need marks as output
    give score on basis of no of correct stage only.
    """

    # Run DeepSeek AI model for evaluation
    evaluation_output = run_ollama_model("deepseek-r1:1.5b", prompt)

    if evaluation_output:
        print("\n**Evaluation Output:**\n", evaluation_output)

        # Extract the final score using regex
        match = re.search(r"Final Score:\s*(\d+)/4", evaluation_output)
        if match:
            total_score = int(match.group(1))
        else:
            total_score = 0

        # Check for irrelevant answers
        irrelevant_keywords = ["i don't know", "anything", "marks", "please"]
        if any(word in student_answer.lower() for word in irrelevant_keywords):
            print("\n⚠️ The student's answer seems irrelevant. Score: 0/4")
            print("Feedback: The answer does not address the question properly.")
        else:
            print(f"\n✅ Final Score: {total_score}/4")

# Example reference answer
reference_answer =  """A search tree is a tree representation of the search process.
The root of the search tree is the root node, which is
corresponding to the initial state.

Search Tree:
Initial state: The root of the search tree is a search node.
Expanding: Applying the Successor Function to the current state, thereby generating a new set of states.
Leaf nodes: The states having no successors.
    """

student_answer = """Search
free
in
AI-
A
Search
free
is
representation
Ам
1
a
graphicd
Pessible states
до
and
actions.
healpy
agent
explore sel
systematically
by
Considering different
Sequences f
decision.
and
finidiy the
optimum
one.
"""

# step-3 :Expanding Applying successor function to the current state
# thereby generating
# step-4 :leaf nodes: the states
# a
# new set of states.
# having no successors.

# Evaluate the student's answer
evaluate_student_answer(reference_answer, student_answer)
