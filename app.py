import chainlit as cl
import os
import logging
from llama_stack_client import Client
from llama_stack_client.types import UserMessage, SystemMessage
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod



# Configure detailed logging output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the model ID matches what your server has registered.
MODEL_ID = os.getenv("INFERENCE_MODEL", "meta-llama/Llama-3.2-11B-Vision-Instruct")

# Base system prompt that all agents will build upon
BASE_PROMPT_TEMPLATE = (
    "You are an expert assistant for the University of North Texas (UNT), powered by advanced AI technology. "
    "Your goal is to provide clear, confident, and structured answers to help UNT students, faculty, staff, and visitors. "
    "When responding to queries:\n\n"
    "### Must Follow Guidelines:\n"
    "1. **Focus on Specificity**:\n"
    "- Ensure that your answer directly addresses the user's query with relevant information.\n"
    "- If the question is about a specific topic (e.g., Computer Science graduate programs), respond with detailed information specific to that program.\n"
    "- Use contextual information to improve relevance and avoid generic responses.\n\n"
 
    "2. **Structured Answers**:\n"
    "- Always respond using bullet points to make your answer easy to read and understand.\n"
    "- Ensure each bullet point provides actionable, useful advice.\n\n"
 
    "3. **Do Not Be Vague**:\n"
    "- Avoid vague phrases like 'check the website for more information.' Instead, provide exact information and actionable next steps based on the user's intent.\n\n"
   
    "4. **Relevant URL Links**:\n"
    "- You MUST PROVIDE https URLs that are directly relevant to the user's query verbatim from the context. For example, if they ask about Computer Science, include links to Computer Science-related resources.\n\n"
   
    "5. **Anticipate User Intent**:\n"
    "- Assume that users are seeking specific, helpful information. Anticipate what they would likely ask next (e.g., application deadlines, course requirements, admission contact information) and provide guidance to help them complete their task.\n\n"
   
    "6. **If the question is related to time period or specific dates**:\n"
    "- Give the response corresponding to current year which is 2025 or if it talks about a specific year give response tailored to it.\n\n"
 
    "7. **Tone**:\n"
    "- Always maintain a confident, helpful, and direct tone. Users should feel that their questions have been fully addressed without confusion.\n"
    "- Avoid expressions of uncertainty such as 'I do not know,' 'it is unclear,' or anything similar.\n\n"

    "8. **Generic Questions**:\n"
    "- If a question is generic (e.g., 'Tell me about UNT'), provide detailed information about UNT's academic programs, student life, campus facilities, research opportunities, and any other relevant aspects."
)

# Instantiate the Llama Stack client with the server URL
client = Client(base_url="http://localhost:8321")

class Agent(ABC):
    """Base class for all specialized agents"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.required_inputs = []
        self.collected_inputs = {}
        self.waiting_for_input = False
        self.current_input_key = None
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the specialized system prompt for this agent"""
        pass
    
    def needs_additional_input(self) -> bool:
        """Check if the agent needs more information from the user"""
        if not self.required_inputs:
            return False
            
        for input_item in self.required_inputs:
            if input_item["key"] not in self.collected_inputs:
                return True
        return False
    
    def process_input(self, message: str) -> Dict[str, Any]:
        """Process the user input and decide what to do next"""
        result = {"type": "response", "content": None, "next_question": None}
        
        # If waiting for a specific input, collect it
        if self.waiting_for_input and self.current_input_key:
            self.collected_inputs[self.current_input_key] = message
            self.waiting_for_input = False
            self.current_input_key = None
        
        # Check if we need more inputs
        if self.needs_additional_input():
            # Find the next required input
            for input_item in self.required_inputs:
                if input_item["key"] not in self.collected_inputs:
                    self.waiting_for_input = True
                    self.current_input_key = input_item["key"]
                    result["type"] = "input_request"
                    result["next_question"] = input_item["question"]
                    break
        else:
            # All inputs collected, ready for final response
            result["type"] = "final_response"
            
        return result
    
    async def get_response(self, user_message: str, attachments=None) -> str:
        """Get a response from the LLM using this agent's specialized prompt"""
        # Build the messages using the typed classes
        system_msg = SystemMessage(content=self.get_system_prompt(), role="system")
        
        # Process attachments if any
        if attachments:
            try:
                # Retrieve image bytes from the first attachment
                image_data = await attachments[0].get_bytes()
                logger.info(f"Received image attachment of size: {len(image_data)} bytes")
                user_message_content = f"{user_message}\n[Image attached]"
                
                # Here you would need to implement actual image handling
                # This depends on how your LLM client handles images
                user_msg = UserMessage(content=user_message_content, role="user")
                # For a vision model, you would add the image data to the message
                
            except Exception as img_err:
                logger.exception("Error retrieving image attachment:")
                user_message_content = f"{user_message}\n[Image attachment error: {str(img_err)}]"
                user_msg = UserMessage(content=user_message_content, role="user")
        else:
            user_msg = UserMessage(content=user_message, role="user")
        
        try:
            # Make the inference request using the chat_completion endpoint
            response = client.inference.chat_completion(
                messages=[system_msg, user_msg],
                model_id=MODEL_ID,
                stream=False
            )
            
            # Extract and log the response content
            reply = response.completion_message.content
            logger.info(f"Inference response from {self.name} agent")
            return reply
        except Exception as e:
            logger.exception("Error during inference:")
            return f"Error: {str(e)}"

    def reset(self):
        """Reset the agent state for a new conversation"""
        self.collected_inputs = {}
        self.waiting_for_input = False
        self.current_input_key = None


class EmailComposeAgent(Agent):
    """Agent specialized in helping with email composition"""
    
    def __init__(self):
        super().__init__(
            name="Email Composer",
            description="Helps compose professional emails for academic settings"
        )
        self.required_inputs = [
            {"key": "recipient_type", "question": "Who are you writing to? (e.g., professor, advisor, administrator)"},
            {"key": "purpose", "question": "What is the main purpose of your email?"},
            {"key": "details", "question": "Any specific details to include in the email?"}
        ]
    
    def get_system_prompt(self) -> str:
        base_prompt = BASE_PROMPT_TEMPLATE
        specialized_instructions = (
            "\n\nYou are now functioning as an Email Composition Assistant. "
            "Help the user draft professional emails for academic settings. "
            "Create emails that are concise, respectful, and clearly communicate the student's needs. "
            "Include all necessary components: greeting, introduction, body, request, gratitude, and signature. "
            "Ensure the tone is appropriate for academic correspondence."
        )
        
        # Add collected inputs to the prompt
        if self.collected_inputs:
            specialized_instructions += "\n\nUser has provided the following details:\n"
            for key, value in self.collected_inputs.items():
                specialized_instructions += f"- {key}: {value}\n"
            
            specialized_instructions += "\nBased on this information, draft a complete, professional email."
        
        return base_prompt + specialized_instructions


class ResearchPaperAgent(Agent):
    """Agent specialized in helping with research papers"""
    
    def __init__(self):
        super().__init__(
            name="Research Paper Assistant",
            description="Helps structure and develop academic research papers"
        )
        self.required_inputs = [
            {"key": "paper_topic", "question": "What is the main topic of your research paper?"},
            {"key": "academic_level", "question": "What is your academic level? (undergraduate, graduate, doctoral)"},
            {"key": "paper_length", "question": "What is the approximate length requirement for your paper?"}
        ]
    
    def get_system_prompt(self) -> str:
        base_prompt = BASE_PROMPT_TEMPLATE
        specialized_instructions = (
            "\n\nYou are now functioning as a Research Paper Assistant. "
            "Help students plan, structure, and develop their academic research papers. "
            "Provide guidance on creating clear thesis statements, organizing sections, integrating sources, "
            "and developing strong arguments. Focus on academic writing conventions and research methodologies."
        )
        
        # Add collected inputs to the prompt
        if self.collected_inputs:
            specialized_instructions += "\n\nUser has provided the following details:\n"
            for key, value in self.collected_inputs.items():
                specialized_instructions += f"- {key}: {value}\n"
            
            specialized_instructions += "\nBased on this information, provide a structured outline and guidance."
        
        return base_prompt + specialized_instructions


class AcademicConceptsAgent(Agent):
    """Agent specialized in explaining academic concepts"""
    
    def __init__(self):
        super().__init__(
            name="Academic Concepts Guide",
            description="Explains complex academic concepts, theories, and works"
        )
        self.required_inputs = [
            {"key": "subject_area", "question": "What subject area are you interested in? (e.g., physics, literature, history)"},
            {"key": "concept", "question": "What specific concept, theory, book, or author would you like explained?"}
        ]
    
    def get_system_prompt(self) -> str:
        base_prompt = BASE_PROMPT_TEMPLATE
        specialized_instructions = (
            "\n\nYou are now functioning as an Academic Concepts Guide. "
            "Provide clear, in-depth explanations of academic concepts, theories, books, and authors. "
            "Focus on explaining complex ideas in an accessible way while maintaining academic accuracy. "
            "Include key points, historical context, significance in the field, and connections to related concepts."
        )
        
        # Add collected inputs to the prompt
        if self.collected_inputs:
            specialized_instructions += "\n\nUser has provided the following details:\n"
            for key, value in self.collected_inputs.items():
                specialized_instructions += f"- {key}: {value}\n"
            
            specialized_instructions += "\nBased on this information, provide a comprehensive explanation."
        
        return base_prompt + specialized_instructions


class RedirectAgent(Agent):
    """Agent specialized in redirecting users to university resources"""
    
    def __init__(self):
        super().__init__(
            name="Resource Redirector",
            description="Guides users to appropriate University resources and websites"
        )
        self.required_inputs = [
            {"key": "resource_type", "question": "What type of resource are you looking for? (e.g., admissions, financial aid, department website)"},
            {"key": "specific_need", "question": "What specific information do you need from this resource?"}
        ]
    
    def get_system_prompt(self) -> str:
        base_prompt = BASE_PROMPT_TEMPLATE
        specialized_instructions = (
            "\n\nYou are now functioning as a University Resource Guide. "
            "Your primary role is to direct users to the most relevant UNT resources and websites. "
            "Provide exact URLs to official UNT pages that address the user's specific needs. "
            "Explain what information they will find at each resource and why it's relevant to their query."
        )
        
        # Add collected inputs to the prompt
        if self.collected_inputs:
            specialized_instructions += "\n\nUser has provided the following details:\n"
            for key, value in self.collected_inputs.items():
                specialized_instructions += f"- {key}: {value}\n"
            
            specialized_instructions += "\nBased on this information, provide relevant UNT resources with URLs."
        
        return base_prompt + specialized_instructions


class GeneralAgent(Agent):
    """General purpose UNT assistant for queries that don't fit specialized categories"""
    
    def __init__(self):
        super().__init__(
            name="General UNT Assistant",
            description="Provides general information about University of North Texas"
        )
        # No required inputs for general agent
        self.required_inputs = []
    
    def get_system_prompt(self) -> str:
        # General agent just uses the base prompt
        return BASE_PROMPT_TEMPLATE


# Agent registry to store and manage our agents
agents = {
    "email": EmailComposeAgent(),
    "research": ResearchPaperAgent(),
    "academic": AcademicConceptsAgent(),
    "redirect": RedirectAgent(),
    "general": GeneralAgent()
}

# Session context to store the active agent for each user session
@cl.cache
def get_session_context():
    return {
        "active_agent": "general",  # Default to general agent
        "conversation_history": []
    }

def determine_agent_type(message: str) -> str:
    """Determine which agent type to use based on the message content"""
    message_lower = message.lower()
    
    if any(word in message_lower for word in ["email", "compose", "write", "draft", "professor"]):
        return "email"
    elif any(word in message_lower for word in ["research", "paper", "thesis", "dissertation", "study", "outline"]):
        return "research"
    elif any(word in message_lower for word in ["concept", "explain", "theory", "book", "author", "academic"]):
        return "academic"
    elif any(word in message_lower for word in ["redirect", "website", "link", "resource", "url", "find"]):
        return "redirect"
    else:
        return "general"


@cl.set_starters
async def set_starters():
    """Define university-related starter suggestions for the welcome screen."""
    return [
        cl.Starter(
            label="Email to Professor",
            message="Help me compose a professional email to my professor requesting an extension for my term paper due to health issues.",
            icon="/public/icons/email.svg",
        ),
        cl.Starter(
            label="Research Paper Assistant",
            message="I need help structuring my research paper on climate change impacts. Can you provide a outline with sections I should include?",
            icon="/public/icons/research.svg",
        ),
        cl.Starter(
            label="Academic related concepts",
            message="Explain the concept of quantum mechanics and its fundamental principles.",
            icon="/public/icons/academic.svg",
        ),
        cl.Starter(
            label="Redirect me to ?",
            message="Where can I find information about graduate admissions requirements for the Computer Science department?",
            icon="/public/icons/url.svg",
        )
    ]


@cl.on_chat_start
async def on_chat_start():
    """Initialize the session and display welcome message"""
    pass


@cl.on_message
async def handle_message(message: cl.Message):
    """
    Handle incoming user messages by:
    1. Determining the appropriate agent
    2. Collecting necessary inputs
    3. Generating responses
    """
    user_input = message.content.strip()
    logger.info(f"Received user input: {user_input}")
    
    # Get the session context and current active agent
    context = get_session_context()
    current_agent_type = context["active_agent"]
    current_agent = agents[current_agent_type]
    
    # Check if this is a new conversation direction
    # Only switch agent if we're not in the middle of collecting inputs
    if not current_agent.waiting_for_input:
        detected_agent_type = determine_agent_type(user_input)
        if detected_agent_type != current_agent_type:
            # Reset the previous agent before switching
            current_agent.reset()
            
            # Switch to the newly detected agent
            context["active_agent"] = detected_agent_type
            current_agent_type = detected_agent_type
            current_agent = agents[current_agent_type]
            logger.info(f"Switched to {current_agent.name} agent")
    
    # Add message to conversation history
    context["conversation_history"].append({"role": "user", "content": user_input})
    
    # Check for attachments
    attachments = getattr(message, "attachments", None)
    
    # Process the input with the current agent
    result = current_agent.process_input(user_input)
    
    if result["type"] == "input_request":
        # Agent needs more information, ask a follow-up question
        await cl.Message(content=result["next_question"]).send()
        # Add this system question to conversation history
        context["conversation_history"].append({"role": "system", "content": result["next_question"]})
    else:
        # Agent has all it needs, generate a response
        # Show thinking indicator
        msg = cl.Message(content="")
        await msg.send()
        
        # Generate response
        response = await current_agent.get_response(user_input, attachments)
        msg.content = response
        # Update the message with the response
        await msg.update()
        
        # Add response to conversation history
        context["conversation_history"].append({"role": "assistant", "content": response})


if __name__ == "__main__":
    cl.run()