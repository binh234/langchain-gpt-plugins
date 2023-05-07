import os
import gradio as gr

from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools import AIPluginTool


def run(prompt, plugin_json, openai_api_key):
    os.environ["OPENAI_API_KEY"] = openai_api_key
    tool = AIPluginTool.from_plugin_url(plugin_json)
    llm = ChatOpenAI(temperature=0, max_tokens=1000)
    tools = load_tools(["requests_all"])
    tools += [tool]
    agent_chain = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        max_tokens_limit=4097,
    )
    return agent_chain.run(prompt)


title = """
<div style="text-align:center;">
  <h1>LangChain + ChatGPT Plugins playground</h1>
  <p>
    This is a demo for the 
    <a href="https://python.langchain.com/en/latest/modules/agents/tools/examples/chatgpt_plugins.html" target="_blank">
    ChatGPT Plugins LangChain
    </a> usecase<br />
    Be aware that it currently only works with plugins that do not require auth.<br />
    Find more plugins <a href="https://www.getit.ai/gpt-plugins" target="_blank">here</a><br />
    Get your OpenAI API key <a href="https://platform.openai.com/account/api-keys" target="_blank">here</a>
  </p>
</div>
"""

with gr.Blocks(css="style.css") as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML(title)
        prompt = gr.Textbox(
            label="Prompt", value="what t shirts are available in klarna?"
        )
        plugin = gr.Textbox(
            label="Plugin json",
            info="You need the .json plugin manifest file of the plugin you want to use. Be aware that it currently only works with plugins that do not require auth.",
            value="https://www.klarna.com/.well-known/ai-plugin.json",
        )
        openai_api_key = gr.Textbox(
            label="OpenAI API Key", info="*required", type="password"
        )
        run_btn = gr.Button("Run")
        response = gr.Textbox(label="Response")
    run_btn.click(fn=run, inputs=[prompt, plugin, openai_api_key], outputs=[response])

demo.queue().launch()
