import time
import random
from .state import State
from langchain.schema import HumanMessage, AIMessage


class Node():
    def __init__(self, name):
        self.name = name

    def process(self, state:State):
        raise NotImplementedError
    
    def __call__(self, state: State):
        res = self.process(state)
        print(f"Node {self.name} processed")
        return res
    

class Supervisor(Node):
    def __init__(self):
        super().__init__("Supervisor")
    
    def process(self, state:State) -> dict:
        # This will act as a system prompt for the llm
        task = state['messages'][0].content if state['messages'] else "No task set."
        thought_prompt = f"Think before responding: {task}"
        state['feedback'] = ''
        state['relevancy'] = 0.0
        state['agent_outputs'] = []
        return {
            "messages": [HumanMessage(role="system", content=f"Iteration {state['iteration']}: {thought_prompt}")]
        }


class Agent(Node):
    def __init__(self, agent_name:str, llm, prompt: str):
        super().__init__(agent_name)
        self.llm = llm
        self.prompt = prompt

    # TODO: Add text to differentiate between different agents
    def process(self, state:State) -> dict:
        # Generate random sleep so that more than 2 requests are not processes in the same time.
        time.sleep(random.randint(1,5))
        task = state['messages'][-1].content
        feedback = state.get('feedback', '')
        resume = state.get('resume', '')
        job_description = state.get('job_description', '')
        if resume and job_description:
            task += f"\nResume: {resume}\nJob Description: {job_description}"
        if self.prompt:
            task += f"\n{self.prompt}"
        if feedback:
            task += f"\nFeedback: {feedback}"
        prompt = f"You are {self.name}.\n{task}"
        response = self.llm(messages=[HumanMessage(role="user", content=prompt)])
        agent_output = AIMessage(role="assistant", content=f"{self.name} response: {response.content}")
        state['agent_outputs'].append(agent_output)
        time.sleep(1)
        return {"messages": [agent_output], "agent_outputs": [agent_output]}
    

class Aggregator(Node):
    def __init__(self, llm):
        super().__init__("Aggregator")
        self.llm = llm

    # TODO: Push evaluators before aggregators and aggregate as per evaluation score weights
    def process(self, state:State) -> dict:
        agent_outputs = state['agent_outputs']
        agent_outputs_text = "\n".join([msg.content for msg in agent_outputs])
        with open("./prompts/aggregator_prompt.txt", "r") as f:
            prompt = f.read()
        review_prompt = (
            f"Agent outputs:\n{agent_outputs_text}\n\n"
            "{prompt}\n"
            "Format: \nFinal Resume: ..."
        )
        response = self.llm(messages=[HumanMessage(role="user", content=review_prompt)])
        content = response.content
        resume = self.parse_final_resume(content)
        # print("Aggregator Resume:", resume)
        state['resume'] = resume
        return {
            "messages": [AIMessage(role="system", content=f"Review result: {content}")],
            "agent_outputs": [AIMessage(role="assistant", content=f"{self.name} response: {content}")],
            "resume": resume
        }
    
    def parse_final_resume(self, content:str) -> str:
        resume = content.replace("Final Resume:", "").strip()
        return resume
    


class Evaluator(Node):
    def __init__(self, llm):
        super().__init__("Evaluator")
        self.llm = llm

    def process(self, state:State) -> dict:
        job_description = state['job_description']
        resume = state.get('resume', '')
        # print("Evaluator Resume:", resume)
        review_prompt = (
            f"Job Description:\n{job_description}\n\n"
            f"Resume:\n{resume}\n\n"
            "Please compare the Job Description with the resume, "
            "and provide feedback to improve the relevancy score"
            "between job description and resume and a relevancy score "
            "based on keywords, qualifications and skills between 0 and 1.\n"
            "Output the float value only for relevancy score.\n"
            "Format:\nFeedback: ...\nRelevancy Score: 0.X"
        )
        response = self.llm(messages=[HumanMessage(role="user", content=review_prompt)])
        content = response.content
        feedback, relevancy = self.parse_feedback_and_relevancy(content)
        state['feedback'] = feedback
        state['relevancy'] = relevancy
        return {
            "messages": [AIMessage(role="system", content=f"Review result: {content}")],
            "feedback": feedback,
            "relevancy": relevancy
        }

    def parse_feedback_and_relevancy(self, content:str) -> float:
        lines = content.splitlines()
        feedback = ''
        relevancy = 0.0
        for line in lines:
            if line.startswith("Feedback:"):
                feedback = line.replace("Feedback:", "").strip()
            elif line.startswith("Relevancy Score:"):
                score_str = line.replace("Relevancy Score:", "").strip()
                try:
                    relevancy = float(score_str)
                except ValueError:
                    print(f"Error: Could not parse relevancy score from: {score_str}")
                    relevancy = 0.0
        return feedback, relevancy


class LoopControlNode(Node):
    def __init__(self, max_iterations: int):
        super().__init__("LoopControl")
        self.max_iterations = max_iterations

    def process(self, state: State) -> dict:
        state['iteration'] += 1  # Increment iteration
        iteration = state['iteration']
        relevancy = state.get('relevancy', 0.0)
        print(f"\nIteration {iteration}, Relevancy Score: {relevancy}")
        if relevancy >= 0.90:
            print("Relevancy score has reached threshold of 90. Terminating the process.")
            state['continue_loop'] = False
        elif iteration >= self.max_iterations:
            print("Maximum number of iterations reached. Terminating the process.")
            state['continue_loop'] = False
        else:
            print("Continuing to the next iteration.")
            state['continue_loop'] = True
        return state