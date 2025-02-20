BASE_INTRO = """User: Hi!

Assistant: Hello! I'm your AI assistant. I'm here to help you with various tasks, such as answering questions, brainstorming ideas, drafting emails, writing code, providing advice, and much more.

User: Why is the sky blue?

Assistant: The sky appears blue due to a phenomenon called Rayleigh scattering, which involves the scattering of sunlight by the molecules and small particles in Earth's atmosphere. Sunlight, which contains a spectrum of colors from red to violet, interacts with the nitrogen and oxygen molecules in the Earth's atmosphere. Among the colors in the visible spectrum, blue and violet have shorter wavelengths. These shorter wavelengths are more likely to be scattered in all directions by the air molecules. This scattering effect is known as Rayleigh scattering. Although violet light is also scattered, our eyes are more sensitive to blue light. Additionally, the scattering of blue light is more efficient, making the sky appear predominantly blue to our eyes.

User: What makes its color variant?

Assistant: The color of the sky can vary depending on the time of day, weather conditions, and atmospheric pollutants. For example, during sunrise and sunset, the sky can appear red or orange due to the longer path the sunlight travels through the atmosphere, scattering more of the blue light away. Moreover, air pollution and cloud cover can also affect the color of the sky, making it appear grey or white.
"""

R1_INTRO = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>\n<answer> answer here </answer>.
User: How long will it take for an object to reach the ground from a height of 50 meters?

Assistant: <think> To calculate the time taken for an object to reach the ground, we can use the formula: t = sqrt(2h/g), where h is the height and g is the acceleration due to gravity. The height is 50 meters and g = 9.81 m/s². </think>
<answer> 3.2 seconds </answer>

User: What is the capital of France?

Assistant: <think> The capital of France is Paris. </think>
<answer> Paris </answer>

User: What is the boiling point of water?

Assistant: <think> The boiling point of water is 100 degrees Celsius at standard atmospheric pressure. </think>
<answer> 100 degrees Celsius </answer>"""

BOX_INTRO = '''
Please reason step by step, and put your final answer within \\boxed{}.
User: How long will it take for an object to reach the ground from a height of 50 meters?

Assistant: To calculate the time taken for an object to reach the ground, we can use the formula: t = sqrt(2h/g), where h is the height and g is the acceleration due to gravity. The height is 50 meters and g = 9.81 m/s². The time taken for the object to reach the ground is \\boxed{3.2 seconds}.

User: What is the capital of France?

Assistant: The capital of France is Paris. The capital of France is \\boxed{Paris}.'''

S1_INTRO = '''
You are a helpful assistant.
User: How long will it take for an object to reach the ground from a height of 50 meters?

Assistant: <think> To calculate the time taken for an object to reach the ground, we can use the formula: t = sqrt(2h/g), where h is the height and g is the acceleration due to gravity. The height is 50 meters and g = 9.81 m/s². </think>
Final Answer: 3.2 seconds

User: What is the capital of France?

Assistant: <think> The capital of France is Paris. </think>
Final Answer: Paris'''