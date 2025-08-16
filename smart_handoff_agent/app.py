# app.py
import os
from dotenv import load_dotenv
import chainlit as cl
from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    handoff,
    AsyncOpenAI,
    Runner,
    set_tracing_disabled,
)

# -------------------------
# Setup
# -------------------------
load_dotenv()
set_tracing_disabled(disabled=True)

# Get API key
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("Please set GEMINI_API_KEY in your .env file")

# External Gemini client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Model configuration
model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

# -------------------------
# Sub-agents
# -------------------------

appointment_agent = Agent(
    name="Appointment Booking Agent",
    instructions="Handle patient requests for booking doctor appointments in a polite and professional manner. Always confirm the date and time before finalizing the booking.",
    model=model
)
custom_appointment_handoff = handoff(
    agent=appointment_agent,
    tool_name_override="custom_appointment_tool",
    tool_description_override="Transfers patient requests for booking a doctor appointment to the Appointment Booking Agent with confirmation of details."
)

lab_report_agent = Agent(
    name="Lab Report Status Agent",
    instructions="Handle patient requests for checking lab test results politely. Always confirm the patient's full name and test date before providing a status update.",
    model=model
)
custom_lab_report_handoff = handoff(
    agent=lab_report_agent,
    tool_name_override="custom_lab_report_tool",
    tool_description_override="Provides polite and accurate updates about the status of a patient's lab test reports, including estimated completion time."
)

general_queries_agent = Agent(
    name="General Queries Agent",
    instructions="Answer general clinic-related questions politely. Provide accurate information about services, working hours, and location.",
    model=model
)
custom_general_queries_handoff = handoff(
    agent=general_queries_agent,
    tool_name_override="custom_general_queries_tool",
    tool_description_override="Handles general clinic-related questions such as location, timings, services offered, and contact details."
)

# -------------------------
# Main triage agent
# -------------------------
triage_agent = Agent(
    name="Healthcare Clinic Agent",
    instructions="""
You are a healthcare triage AI.
Read the patient's request carefully and decide which agent should handle it:
- If the request is for booking a doctor appointment → Forward to the Appointment Booking Agent.
- If the request is for checking the status of a lab report or test result → Forward to the Lab Report Status Agent.
- If the request is for general clinic information → Forward to the General Queries Agent.

Always respond politely in English.
Always use the correct tool call to forward the request — do not give a plain text reply.
""",
    handoffs=[custom_appointment_handoff, custom_lab_report_handoff, custom_general_queries_handoff],
    model=model
)

# -------------------------
# Chainlit app (clean UI)
# -------------------------

@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    await cl.Message(
        content="👩‍⚕️ **Welcome to Healthcare Clinic AI Assistant**\n\nI can help you with:\n- Booking Appointments 🗓️\n- Checking Lab Reports 🧪\n- General Clinic Info ℹ️\n\nHow may I assist you today?"
    ).send()

@cl.on_message
async def handle_message(message: cl.Message):
    # Run triage agent
    result = await Runner.run(triage_agent, input=message.content)

    # Save history (but don't show explicitly)
    history = cl.user_session.get("history", [])
    history.append({"user": message.content, "bot": result.final_output})
    cl.user_session.set("history", history)

    # Show only AI reply (clean chat)
    await cl.Message(content=result.final_output).send()
