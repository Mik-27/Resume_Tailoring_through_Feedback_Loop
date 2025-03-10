from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from langchain.schema import HumanMessage, AIMessage

from src.nodes import *
from src.state import State
from src.util import next_node

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

class ResumeAgent():
    def __init__(self, query="Tailor the provided resume to the job description according to the given goal and instructions, also consider feedback if available.", max_iterations=5, time_limit=500):
        self.max_iterations = max_iterations
        self.time_limit = time_limit
        self.query = query
        self.gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.7)
        self.evaluator = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.2)
        self.aggregator = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.5)


    def _get_prompts(self):
        """
        Get the prompts for the agents
        """
        prompts = []
        with open("./prompts/resume_p1_impact.txt") as f:
            prompts.append(f.read())
        with open("./prompts/resume_p2_skills.txt") as f:
            prompts.append(f.read())
        with open("./prompts/resume_p3_industry.txt") as f:
            prompts.append(f.read())

        return prompts
    

    def _build_graph(self, prompts):
        """
        Build the state graph for the resume review process
        """

        builder = StateGraph(State)

        nodes = {
            "Supervisor": Supervisor(),
            "Agent1": Agent("Agent 1 - Impact", self.gemini, prompts[0]),
            "Agent2": Agent("Agent 2 - Skills", self.gemini, prompts[1]),
            "Agent3": Agent("Agent 3 - Industry", self.gemini, prompts[2]),
            "Aggregator": Aggregator(self.aggregator),
            "Evaluator": Evaluator(self.evaluator),
            "loop_control": LoopControlNode(self.max_iterations),
        }

        for name, node in nodes.items():
            builder.add_node(name, node)

        # Define the flow
        builder.add_edge(START, "Supervisor")

        # Edges from Supervisor to Agents
        builder.add_edge("Supervisor", "Agent1")
        builder.add_edge("Supervisor", "Agent2")
        builder.add_edge("Supervisor", "Agent3")

        # Edges from Agents to Aggregator
        builder.add_edge("Agent1", "Aggregator")
        builder.add_edge("Agent2", "Aggregator")
        builder.add_edge("Agent3", "Aggregator")

        # Edge from Aggregator to Evaluator
        builder.add_edge("Aggregator", "Evaluator")

        # From Reviewer to Loop Control Node
        builder.add_edge("Evaluator", "loop_control")

        # Conditionally decide the next node from Loop Control Node
        builder.add_conditional_edges("loop_control", next_node)

        graph = builder.compile()
        return graph
    

    def _set_state(self, query, resume, job_description):
        """
        Set the initial state for the resume review process
        """
        initial_state = State(
            messages=[HumanMessage(role="user", content=query)],
            iteration=0,
            resume=resume,
            job_description=job_description,
            agent_outputs=[],
            relevancy=0.0,
            continue_loop=True
        )
        return initial_state
    
    def run(self, resume, job_description):
        """
        Run the resume review process
        """
        prompts = self._get_prompts()
        graph = self._build_graph(prompts)
        state = self._set_state(self.query, resume, job_description)
        final_state = graph.invoke(state, {"recursion_limit": self.time_limit})
        return final_state
    

if __name__ == "__main__":
    agent = ResumeAgent()
    query = "Tailor the provided resume to the job description according to the given goal and instructions, also consider feedback if available."
    
    with open("./sample_data/resume.txt", "r") as f:
        resume = f.read()
    with open("./sample_data/jd.txt", "r") as f:
        job_description = f.read()
    
    res = agent.run(resume, job_description)
    print(res['messages'][-1].content)