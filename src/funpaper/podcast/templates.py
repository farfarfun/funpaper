from langchain_core.prompts import ChatPromptTemplate

# Templates
plan_prompt = ChatPromptTemplate.from_template("""You are a very clever planner of podcast scripts. You will be given the text of a research paper, and your task will be to generate a plan for a podcast involving 3 persons discussing about the content of the paper in a very engaging, interactive and enthusiastic way. The plan will be structured using titles and bullet points only. The plan for the podcast should follow the structure of the paper. The podcast involves the following persons:
- The host: he will present the paper and its details in a very engaging way. very professional, friendly, warm and enthusiastic.
- The learner: he will ask clever and significative questions about the paper and its content. he is curious and funny.
- The expert: he will provide deep insights, comments and details about the content of the paper and other related topics. he talks less than the two other and his interventions are more profound and detailed.
Example of a structure for the podcast:
# Title: title of the podcast
# Section 1: title of section 1
- bullet point 1
- bullet point 2
- bullet point 3
...
- bullet point n
# Section 2: title of section 2
- bullet point 1
- bullet point 2
- bullet point 3
...
- bullet point n
# Section 3: title of section 3
...
# Section n: title of section n
- bullet point 1
- bullet point 2
- bullet point 3
...
- bullet point n
The paper: {paper}
The podcast plan in titles and bullet points:""")

discuss_prompt_template = ChatPromptTemplate.from_template("""You are a very clever scriptwriter of podcast discussions. You will be given a plan for a section of the middle of a podcast that already started involving 3 persons discussing about the content of a research paper. Your task will be to generate a brief dialogue for the podcast talking about the given section, do not include voice effects, and do not make an introduction. The dialogue should be engaging, interactive, enthusiastic and have very clever transitions and twists. The dialogue should follow the structure of the plan. The podcast involves the following persons:
- The host: he will present the paper and its details in a very engaging way. very professional, friendly, warm and enthusiastic.
- The learner: he will ask clever and significative questions about the paper and its content. he is curious and funny.
- The expert: he will provide deep insights, comments and details about the content of the paper and other related topics. he talks less than the two other and his interventions are more profound and detailed.
Dialogue example 1:
Host: Let's continue with the second section of the paper ... 
Learner: I have a question about ...
Expert: I would like to add ... 
Dialogue example 2:
Host: Now, let's move on to the next section ...
Expert: I think that ...
Learner: I have a question about ...
Expert: I would like to add ...
Dialogue example 3:
Learner: Should we move on to the next section?
Host: Yes, let's move on to the next section ...
Expert: I think that ...
Section plan: {section_plan}
Previous dialogue (to avoid repetitions): {previous_dialogue}
Additional context:{additional_context}
Brief section dialogue:""")

initial_dialogue_prompt = ChatPromptTemplate.from_template("""You are a very clever scriptwriter of podcast introductions. You will be given the title of a paper and a brief glimpse of the content of a research paper. Avoid using sound effects, only text. Avoid finishing with the host, finish the dialogue with the expert. Your task will be to generate an engaging and enthusiastic introduction for the podcast. The introduction should be captivating, interactive, and should make the listeners eager to hear the discussion. The introduction of the podcast should have 3 interactions only. The podcast involves the following persons:
- The host: he will present the paper and its details in a very engaging way. very professional, friendly, warm and enthusiastic.
- The learner: he will ask clever and significative questions about the paper and its content. he is curious and funny.
- The expert: he will provide deep insights, comments and details about the content of the paper and other related topics. he talks less than the two other and his interventions are more profound and detailed.
Introduction example 1:
Host: Welcome to our podcast, today we will be discussing the paper ...
Learner: I am very curious about ...
Expert: I think that ...
Introduction example 2:
Host: Hello everyone, today we have a very interesting paper to discuss ...
Expert: I would like to add ...
Learner: I have a question about ...
Content of the paper: {paper_head}
Brief 3 interactions introduction:""")

enhance_prompt = ChatPromptTemplate.from_template("""You are a very clever scriptwriter of podcast discussions. You will be given a script for a podcast involving 3 persons discussing about the content of a research paper. Your task will be to enhance the script by removing audio effects mentions and reducing repetition and redundancy. Don't mention sound effects, laughing, chuckling or any other audio effects between brackets. The script should only contain what the persons are saying and not what are they doing or how they are saying it. Enhance the transitions and the twists, and reduce repetition and redundancy.
The draft script{draft_script}
The enhanced script:""")
