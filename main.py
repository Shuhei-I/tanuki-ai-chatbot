import os
import streamlit as st
import tiktoken
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# 環境変数の読み込み
load_dotenv()

# チャット履歴の初期化２
if "messages" not in st.session_state:
    st.session_state.messages = ConversationBufferWindowMemory(return_messages=True,k=3)

# ページ内容
        
st.title("たぬき会話AI")
st.subheader('あなたの要望に :orange[たぬき] が答えます！', divider='rainbow')

messages = st.container(height=300) # メッセージ欄の大きさ

# 入力欄
prompt = st.chat_input("なにかしゃべりかけてよ！")
# 入力されたら
if prompt :
    # ユーザー入力処理
    # 入力内容をst.session_state.messagesに追加し、表記する
    # st.session_state.messages.append({"role":"user","content": prompt})
    messages.chat_message("user").write(prompt)
    
    # 返答処理
    with messages.chat_message("🦝",avatar="statics/images/tanuki.jpg"):
        
        # テンプレートの準備
        template = """あなたは「たぬき」という可愛いケーキです。言葉遣いはフレンドリーで、語尾に「たぬー！」をつけます。
        
        会話内容：{question}
        
        """
        # プロンプトテンプレートの準備
        prompt_template = PromptTemplate(
            input_variables=["question"],
            template=template
        )

        # 会話履歴を用いた会話応答
        llm = ChatOpenAI(model_name="gpt-4", temperature = 0)
        conversation = LLMChain(
            llm=llm,
            prompt=prompt_template,
            memory = st.session_state.messages,
            verbose=True
        )

        response = conversation.run(prompt)
        

        st.write(response)
    
    #トークン数を計算
    enc = tiktoken.get_encoding("gpt2")
    input_tokens =  enc.encode(prompt)
    output_tokens = enc.encode(response)
    tokens = input_tokens + output_tokens
    print("--------------------------------------")
    print(f"TokenResult:input({len(input_tokens)}),output({len(output_tokens)}),total({len(tokens)})")
    print("--------------------------------------")

    # 会話内容をsessionの変数に保存
    # st.session_state.messages.save_context({"input": prompt}, {"output": response})
    print(st.session_state.messages.load_memory_variables({}))

st.caption('あくまでも :orange[たぬきの意見] だから真面目にとらえないでね！')
