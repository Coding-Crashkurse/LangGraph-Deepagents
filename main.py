import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from deepagents.graph import create_deep_agent
from deepagents.middleware.subagents import CompiledSubAgent

load_dotenv()

model = ChatOpenAI(model="gpt-4o", temperature=0)


def run_example_1_simple_chat():
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Simple Chat")
    print("=" * 60)

    agent = create_deep_agent(
        model=model,
        system_prompt="You are a helpful assistant."
    )

    query = "What is the capital of France? Answer only with the city name."
    print(f"\nUser: {query}")

    response = agent.invoke({
        "messages": [HumanMessage(content=query)]
    })

    print(f"\nAgent response: {response['messages'][-1].content}")


def run_example_2_complex_no_subagent():
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Complex Task (Planning YES, Delegation NO)")
    print("=" * 60)

    agent = create_deep_agent(
        model=model,
        system_prompt="You are an organized event planner."
    )

    query = """
    I am planning a small breakfast for 2 people.
    Please provide:
    1. A list with 3 breakfast options.
    2. A shopping list for option 1.
    """
    print(f"\nUser: {query}")

    response = agent.invoke({
        "messages": [HumanMessage(content=query)]
    })

    print(f"\nAgent response (excerpt): {response['messages'][-1].content[:150]}...")


def run_example_3_custom_subagent():
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Delegation to a Custom Sub-Agent")
    print("=" * 60)

    math_expert_graph = create_agent(
        model=model,
        tools=[],
        system_prompt="You are an eccentric math professor. Start every sentence with 'Eureka!'."
    )

    math_subagent_config = CompiledSubAgent(
        name="math-professor",
        description="An expert in complex mathematical proofs.",
        runnable=math_expert_graph
    )

    orchestrator = create_deep_agent(
        model=model,
        subagents=[math_subagent_config],
        system_prompt="You are a project manager. If anything involves math, delegate immediately to the 'math-professor'."
    )

    query = "Explain briefly why 1+1=2."
    print(f"\nUser: {query}")

    response = orchestrator.invoke({
        "messages": [HumanMessage(content=query)]
    })

    print(f"\nAgent response: {response['messages'][-1].content}")


if __name__ == "__main__":
    run_example_1_simple_chat()
    time.sleep(2)
    run_example_2_complex_no_subagent()
    time.sleep(2)
    run_example_3_custom_subagent()
