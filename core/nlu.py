# core/nlu.py

import sys
import json
import time
import requests
import os
from typing import Optional, List, Dict, Any



class HestiaNLU:
    """Natural language understanding using Ollama with structured JSON output."""

    def __init__(self, model: str = "mistral", host: str = "localhost",
                 port: int = 11434, prompt_path: str = "config/nlu_prompt.txt", providers: list = None):
        """Initialize LLM providers and load system prompt from file."""
        self.model = model
        self.base_url = f"http://{host}:{port}"
        self.temperature = 0.1
        self.max_tokens = 150
        self.system_prompt = self._load_prompt(prompt_path)
        self.providers = providers or [
            {"name": "ollama", "model": self.model, "host": host, "port": port}
        ]
        self._memory = None

    def _load_prompt(self, path: str) -> str:
        """Load system prompt and few-shot examples from file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                print("NLU prompt loaded successfully", file=sys.stderr)
                return content
        except Exception as e:
            print(f"WARNING: Using fallback NLU prompt: {e}", file=sys.stderr)
            # Minimal fallback prompt
            return (
                "You are Hestia, a warm, playful, affectionate personal assistant.\n"
                "Always respond with valid JSON: {\"intent\": \"chat\", \"entities\": {}, \"response\": \"...\", \"confidence\": 0.9}\n"
                "Valid intents: chat, get_time, get_date, get_weather, set_reminder, open_app, take_note, save_name, get_user_info, get_history, get_notes, set_preference\n"
            )

    def _build_prompt(self, text: str, context: Optional[List[Dict[str, Any]]] = None) -> str:
        """Construct full prompt with system prompt, context, user facts, and user input."""
        prompt = self.system_prompt + "\n"

        if context and len(context) > 0:
            recent = context[-3:] if len(context) > 3 else context
            prompt += "Recent conversation:\n"
            for item in recent:
                q = item.get('query', '')
                r = item.get('response', '')
                prompt += f"User said: '{q}' | You responded: '{r}'\n"

        # Inject known user facts
        facts_context = ""
        try:
            # memory is not directly available on NLU — skip silently if not injected
            if hasattr(self, "_memory") and self._memory is not None:
                facts_context = self._memory.get_top_facts_for_context(limit=5)
        except Exception:
            pass
        if facts_context:
            prompt += f"\n{facts_context}\n"

        prompt += f"""
        User: {text}

        You MUST respond with STRICT JSON ONLY.
        NO text before or after.
        NO explanations.

        Format:
        {{
        "intent": "...",
        "entities": {{}},
        "response": "...",
        "confidence": 0.0
        }}
        """
        return prompt

    def set_memory(self, memory) -> None:
        """Inject memory reference so NLU can include user facts in prompts."""
        self._memory = memory
    
    def _health_check(self):
        """Light ping only — don't run inference."""
        try:
            r = requests.get(self.base_url, timeout=3)
            return r.status_code == 200
        except:
            return False
        return True

    def understand(self, text, context=None):
        """Parse user input — one health check, then retry real calls only."""
        if not self._health_check():
            print("[NLU] Ollama unreachable", file=sys.stderr)
            return {"intent": "chat", "entities": {}, "response": "My backend isn't responding right now.", "confidence": 0.0}

        prompt = self._build_prompt(text, context)
        for attempt in range(5):
            print(f"[NLU] Attempt {attempt + 1}/5", file=sys.stderr)
            response = self._call_llm(prompt)
            if response:
                parsed = self._parse_response(response)
                print(f"[NLU PARSED]: {parsed}", file=sys.stderr)
                if parsed.get("intent"):
                    return parsed
            time.sleep(1.5)

        return {"intent": "chat", "entities": {}, "response": "Sorry, I had trouble understanding that.", "confidence": 0.5}

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Try each provider in order. Return first successful response text, or None if all fail."""
        for provider in self.providers:
            name = provider.get("name", "ollama")
            try:
                if name == "ollama":
                    result = self._call_ollama_provider(provider, prompt)
                elif name == "anthropic":
                    result = self._call_anthropic_provider(provider, prompt)
                elif name == "gemini":
                    result = self._call_gemini_provider(provider, prompt)
                else:
                    print(f"[NLU] Unknown provider '{name}', skipping.", file=sys.stderr)
                    continue
                if result:
                    return result
            except Exception as e:
                print(f"[NLU] Provider '{name}' failed: {e}", file=sys.stderr)
                continue
        return None

    def _call_ollama_provider(self, provider: dict, prompt: str) -> Optional[str]:
        """Call Ollama API using provider config dict."""
        host = provider.get("host", "127.0.0.1")
        port = provider.get("port", 11434)
        model = provider.get("model", self.model)
        url = f"http://{host}:{port}/api/generate"
        body = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens}
        }
        try:
            resp = requests.post(url, json=body, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            response_text = data.get("response", "")
            if not response_text:
                time.sleep(2)
                return None
            return response_text
        except Exception as e:
            print(f"[NLU ERROR] {e}", file=sys.stderr)
            return None

    def _call_anthropic_provider(self, provider: dict, prompt: str) -> Optional[str]:
        """Call Anthropic API using the anthropic SDK."""
        try:
            import anthropic
        except ImportError:
            print("[NLU] anthropic package not installed.", file=sys.stderr)
            return None
        api_key = provider.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("[NLU] No ANTHROPIC_API_KEY found.", file=sys.stderr)
            return None
        model = provider.get("model", "claude-haiku-4-5-20251001")
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=200,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text if message.content else None

    def _call_gemini_provider(self, provider: dict, prompt: str) -> Optional[str]:
        """Call Google Gemini API using google-generativeai SDK."""
        try:
            import google.generativeai as genai
        except ImportError:
            print("[NLU] google-generativeai package not installed.", file=sys.stderr)
            return None
        api_key = provider.get("api_key") or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            print("[NLU] No GEMINI_API_KEY found.", file=sys.stderr)
            return None
        model_name = provider.get("model", "gemini-1.5-flash")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=self.system_prompt
        )
        response = model.generate_content(prompt)
        return response.text if response.text else None

    def _call_ollama(self, prompt):
        """Real generate call — no health check, no WSL subprocess."""
        url = f"{self.base_url}/api/generate"
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens}
        }
        try:
            resp = requests.post(url, json=body, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            response_text = data.get("response", "")
            if not response_text:
                time.sleep(2)
                return None
            return response_text
        except Exception as e:
            print(f"[NLU ERROR] {e}", file=sys.stderr)
            return None

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Extract and validate JSON from LLM response."""
        # Check for JSON presence
        if "{" not in response:
            print("Invalid JSON structure: no opening brace found", file=sys.stderr)
            return {
                "intent": "chat",
                "entities": {},
                "response": response,
                "confidence": 0.5
            }
        

        # Strip code fences
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]

        if response.strip().startswith('"') and response.strip().endswith('"'):
            response = response.strip('"')

        # Extract first JSON object
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end > start:
            json_str = response[start:end]
        else:
            json_str = response

        try:
            obj = json.loads(json_str)
            
            # Validate required fields
            intent = obj.get("intent", "chat")
            entities = obj.get("entities", {})
            response_text = obj.get("response", response)
            confidence = obj.get("confidence", 0.5)
            
            # Ensure correct types
            if not isinstance(intent, str):
                intent = "chat"
            if not isinstance(entities, dict):
                entities = {}
            if not isinstance(response_text, str):
                response_text = str(response_text)
            if not isinstance(confidence, (int, float)):
                confidence = 0.5
            else:
                confidence = float(confidence)
            
            return {
                "intent": intent,
                "entities": entities,
                "response": response_text,
                "confidence": confidence
            }
        except Exception as e:
            print(f"Invalid JSON structure: {e}", file=sys.stderr)
            return {
                "intent": "chat",
                "entities": {},
                "response": response,
                "confidence": 0.5
            }