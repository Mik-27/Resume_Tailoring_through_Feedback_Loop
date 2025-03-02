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
        task = state['messages'][0].content if state['messages'] else "No task set."
        thought_prompt = f"Think before responding: {task}"
        expected_result = f"{thought_prompt}"
        state['expected_result'] = expected_result
        state['feedback'] = ''
        state['evaluation'] = 0.0
        state['agent_outputs'] = []
        return {
            "messages": [HumanMessage(role="system", content=f"Iteration {state['iteration']}: {thought_prompt}")],
            "expected_result": expected_result
        }
    
class Agent(Node):
    def __init__(self, agent_name:str, llm):
        super().__init__(agent_name)
        self.llm = llm

    def process(self, state:State) -> dict:
        task = state['messages'][-1].content
        feedback = state.get('feedback', '')
        if feedback:
            task += f"\nFeedback: {feedback}"
        prompt = f"You are {self.name}. Please perform the following task: {task}"
        response = self.llm(messages=[HumanMessage(role="user", content=prompt)])
        agent_output = AIMessage(role="assistant", content=f"{self.name} response: {response.content}")
        state['agent_outputs'].append(agent_output)
        return {"messages": [agent_output], "agent_outputs": [agent_output]}
    
class Evaluator(Node):
    def __init__(self, llm):
        super().__init__("Evaluator")
        self.llm = llm

    def process(self, state:State) -> dict:
        expected_result = state['expected_result']
        agent_outputs = state.get('agent_outputs', [])
        agent_outputs_text = "\n".join([msg.content for msg in agent_outputs])
        review_prompt = (
            f"Expected result:\n{expected_result}\n\n"
            f"Agent outputs:\n{agent_outputs_text}\n\n"
            "Please compare the expected result with the agent outputs, "
            "and provide feedback and an evaluation score between 0 and 1.\n"
            "Format:\nFeedback: ...\nEvaluation Score: 0.X"
        )
        response = self.llm(messages=[HumanMessage(role="user", content=review_prompt)])
        content = response.content
        feedback, evaluation = self.parse_feedback_and_evaluation(content)
        state['feedback'] = feedback
        state['evaluation'] = evaluation
        return {
            "messages": [AIMessage(role="system", content=f"Review result: {content}")],
            "feedback": feedback,
            "evaluation": evaluation
        }

    def parse_feedback_and_evaluation(self, content:str) -> float:
        lines = content.splitlines()
        feedback = ''
        evaluation = 0.0
        for line in lines:
            if line.startswith("Feedback:"):
                feedback = line.replace("Feedback:", "").strip()
            elif line.startswith("Evaluation Score:"):
                score_str = line.replace("Evaluation Score:", "").strip()
                try:
                    evaluation = float(score_str)
                except ValueError:
                    evaluation = 0.0
        return feedback, evaluation


class LoopControlNode(Node):
    def __init__(self, max_iterations: int):
        super().__init__("LoopControl")
        self.max_iterations = max_iterations

    def process(self, state: State) -> dict:
        state['iteration'] += 1  # Increment iteration
        iteration = state['iteration']
        evaluation = state['evaluation']
        print(f"Iteration {iteration}, Evaluation Score: {evaluation}")
        if evaluation >= 1:
            print("Evaluation score has reached 1 or higher. Terminating the process.")
            state['continue_loop'] = False
        elif iteration >= self.max_iterations:
            print("Maximum number of iterations reached. Terminating the process.")
            state['continue_loop'] = False
        else:
            print("Continuing to the next iteration.")
            state['continue_loop'] = True
        return state