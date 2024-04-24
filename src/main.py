from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process
import os

## test 1
# email = "nigerian prince sending some gold"

## test 2
email = "hey, your neighbor john here , your house seems to be on fire, you should probably check it out"

classifier = Agent(
    role="email classifier",
    goal="accurate classify emails based on their importance. give every email one of these ratings: important, casual or spam",
    backstory="you are an ai assistant whose only job is to classify emails accurately and honestly. do not be afraid to give emails bad rating if they are not important. your job is to help the user manage their inbox.",
    verbose=True,
    allow_delegation=False,
    # llm=model,
)

responder = Agent(
    role="email responder",
    goal="based on the importance of the email, write a concise and simple response to the email. if the email is rated important write a formal response, if email is rated casual write a casual response, if email is rated spam, do not respond. no matter what, be very concise.",
    backstory="you are an ai assistant whose only job is to write short responses to emails based on their importance. the importance will be provided to you by the classifier agent.",
    verbose=True,
    allow_delegation=False,
    # llm=model,
)

classify_email = Task(
    description=f"Classify the email: {email}",
    agent=classifier,
    expected_output="One of these ratings: important, casual or spam",
)

respond_to_email = Task(
    description=f"Respond to the email: {email} based on the importance provided by the  classifier agent.",
    agent=responder,
    expected_output="a very concise response to the email based on the importance provided by the classifier agent.",
)

crew = Crew(
    agents=[classifier, responder],
    tasks=[classify_email, respond_to_email],
    verbose=2,
    process=Process.sequential,
)

output = crew.kickoff()
print(output)
# save output in a file with the name of the email without spaces
email_name = email.replace(" ", "_")
output_file = f"output/{email_name}.txt"
# Ensure the output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, "w") as f:
    f.write(str(output))
print(f"Output saved in {output_file}")
