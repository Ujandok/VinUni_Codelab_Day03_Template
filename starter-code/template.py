"""
Lab #3: Baseline Chatbot vs ReAct Agent
"""

import json
import os
import re

from tools import TOOL_DEFINITIONS, TOOL_MAP, get_flight_info, get_weather_forecast

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""


def call_gemini(system_prompt: str, user_prompt: str) -> str:
    """Gọi Gemini 1 lượt (cần GEMINI_API_KEY)."""
    import google.generativeai as genai

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=system_prompt)
    response = model.generate_content(
        user_prompt,
        generation_config={"temperature": 0.3, "stop_sequences": ["Observation:"]},
    )
    return response.text


class MockLLM:
    """LLM giả lập để chạy/test offline khi không có API key."""

    def __call__(self, system_prompt, scratchpad):
        question = scratchpad.splitlines()[0]
        codes = re.findall(r"\b(HAN|SGN|DAD)\b", question.upper())
        origin = codes[0] if codes else "HAN"
        dest = codes[-1] if codes else "SGN"
        want_flight = bool(re.search(r"chuyến bay|vé|bay", question, re.I)) and len(codes) >= 2
        want_weather = bool(re.search(r"thời tiết|mặc gì|nhiệt độ", question, re.I))

        if not (want_flight or want_weather):
            return ("Thought: Câu hỏi ngoài phạm vi 2 công cụ.\n"
                    "Final Answer: Tôi chỉ hỗ trợ tìm chuyến bay và tra cứu thời tiết. "
                    "Vui lòng liên hệ tổng đài CSKH để được giải đáp.")

        if want_flight and "get_flight_info" not in scratchpad:
            return (f"Thought: Tra cứu chuyến bay {origin} - {dest}.\n"
                    'Action: {"name": "get_flight_info", "args": '
                    f'{{"origin": "{origin}", "destination": "{dest}", '
                    f'"max_price": {self._max_price(question)}}}}}')

        if want_weather and "get_weather_forecast" not in scratchpad:
            return (f"Thought: Tra cứu thời tiết {dest}.\n"
                    'Action: {"name": "get_weather_forecast", "args": '
                    f'{{"city_code": "{dest}"}}}}')

        return "Thought: Đã đủ dữ liệu.\nFinal Answer: " + self._summarize(scratchpad)

    @staticmethod
    def _max_price(text):
        m = re.search(r"(\d+(?:[.,]\d+)?)\s*(triệu|k)", text, re.I)
        if not m:
            return 5000000
        unit = 1_000_000 if m.group(2).lower() == "triệu" else 1_000
        return int(float(m.group(1).replace(",", ".")) * unit)

    @staticmethod
    def _summarize(scratchpad):
        parts = []
        for raw in re.findall(r"Observation:\s*(.+)", scratchpad):
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(data, list):
                if data:
                    best = min(data, key=lambda f: f["price_vnd"])
                    parts.append(f"Có {len(data)} chuyến phù hợp, rẻ nhất là "
                                 f"{best['airline']} {best['flight_number']} lúc "
                                 f"{best['departure_time']}, giá {best['price_vnd']:,} VND.")
                else:
                    parts.append("Không tìm thấy chuyến bay nào phù hợp với yêu cầu.")
            elif isinstance(data, dict) and "recommendation" in data:
                parts.append(f"Thời tiết {data['city']}: {data['temperature_c']}°C, "
                             f"{data['condition']}. Gợi ý: {data['recommendation']}")
        return " ".join(parts)


def default_llm():
    return call_gemini if os.getenv("GEMINI_API_KEY") else MockLLM()


class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""

    def query(self, user_input: str) -> str:
        if os.getenv("GEMINI_API_KEY"):
            return call_gemini("Bạn là chatbot tư vấn du lịch, không có tool tra cứu.",
                               user_input)
        return ("[Chatbot Baseline] Tôi không có quyền truy cập hệ thống đặt vé hay dữ liệu "
                "thời tiết, nên không kiểm tra được chuyến bay cụ thể. Chặng HAN-SGN thường "
                "khoảng 1-3 triệu VND, miền Nam nóng nên mặc đồ mát. (Đây là phỏng đoán, "
                f"không phải dữ liệu thật.) Câu hỏi: {user_input}")


class ReActAgent:
    """ReAct Agent có sử dụng Thought-Action-Observation Loop"""

    def __init__(self, max_iterations: int = 5, llm=None):
        self.max_iterations = max_iterations
        self.trace = []
        self.llm = llm or default_llm()
        self.system_prompt = SYSTEM_PROMPT.format(
            tools=json.dumps(TOOL_DEFINITIONS, ensure_ascii=False, indent=2))

    def parse_action(self, text):
        """Tách JSON sau 'Action:'. Trả về (action, error)."""
        if "Action:" not in text:
            return None, None
        m = re.search(r"Action:\s*(\{.*\})", text, re.DOTALL)
        if not m:
            return None, "Invalid JSON format"
        try:
            action = json.loads(m.group(1))
        except json.JSONDecodeError:
            return None, "Invalid JSON format"
        if "name" not in action:
            return None, "Action thiếu trường 'name'"
        action.setdefault("args", {})
        return action, None

    def execute_tool(self, action):
        """Trap 1: chuẩn hoá tên tool trước khi tra TOOL_MAP."""
        name = str(action["name"]).strip().lower()
        if name not in TOOL_MAP:
            return {"error": f"Tool '{name}' không tồn tại"}
        try:
            return TOOL_MAP[name](**action["args"])
        except Exception as e:
            return {"error": f"Lỗi khi gọi {name}: {e}"}

    def run(self, user_input: str) -> dict:
        # TODO 1: lưu lịch sử
        self.trace = [{"step": "init", "user_input": user_input}]
        scratchpad = f"Câu hỏi của khách hàng: {user_input}\n"
        iteration = 0
        errors = 0

        # TODO 2: vòng lặp
        while iteration < self.max_iterations:
            iteration += 1
            output = self.llm(self.system_prompt, scratchpad)

            # TODO 3: phân tích Thought / Action
            thought = re.search(r"Thought:\s*(.+)", output)
            final = re.search(r"Final Answer:\s*(.+)", output, re.DOTALL)
            action, error = self.parse_action(output)

            step = {"iteration": iteration,
                    "thought": thought.group(1).strip() if thought else None}

            if final and not action and not error:
                step["final_answer"] = final.group(1).strip()
                self.trace.append(step)
                return {"status": "success", "answer": step["final_answer"],
                        "trace": self.trace}

            # TODO 4: thực thi tool
            if action:
                observation = self.execute_tool(action)
                step["action"] = action
                errors = errors + 1 if isinstance(observation, dict) and "error" in observation else 0
            else:
                observation = error or "Thiếu Action hoặc Final Answer"
                errors += 1

            # TODO 5: ghi Observation rồi lặp tiếp
            step["observation"] = observation
            self.trace.append(step)
            scratchpad += f"{output}\nObservation: {json.dumps(observation, ensure_ascii=False)}\n"

            if errors >= 2:
                return {"status": "tool_error",
                        "answer": "Xin lỗi, hệ thống chưa tra cứu được dữ liệu bạn yêu cầu.",
                        "trace": self.trace}

        return {"status": "max_iterations_reached",
                "answer": "Không thể hoàn thành trong số bước tối đa.",
                "trace": self.trace}


def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
