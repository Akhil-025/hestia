import os
import sys
import importlib.util
from typing import Optional

class SkillLoader:
    """Scans the skills/ directory, loads each skill module, and registers
    their intents into the NLU prompt and the actions dispatcher."""

    SKILL_REGISTRY: dict[str, dict] = {}
    # Maps intent name -> {"execute": Callable, "examples": list[str], "description": str}

    def __init__(self, skills_dir: str = "skills"):
        """
        Args:
          skills_dir: Path to the skills directory relative to project root.
        """
        self.skills_dir = skills_dir
        self.loaded: list[str] = []

    def load_all(self) -> int:
        """
        Scan skills_dir for .py files (excluding __init__.py and base.py),
        import each, call its register() function, and store results in SKILL_REGISTRY.
        Returns count of successfully loaded skills.
        """
        count = 0
        if not os.path.isdir(self.skills_dir):
            print(f"[Skills] Directory '{self.skills_dir}' not found, skipping.", file=sys.stderr)
            return 0

        for filename in sorted(os.listdir(self.skills_dir)):
            if not filename.endswith(".py"):
                continue
            if filename in ("__init__.py", "base.py"):
                continue
            skill_path = os.path.join(self.skills_dir, filename)
            skill_name = filename[:-3]
            try:
                spec = importlib.util.spec_from_file_location(f"skills.{skill_name}", skill_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if not hasattr(module, "register"):
                    print(f"[Skills] '{filename}' has no register() function, skipping.", file=sys.stderr)
                    continue

                registration = module.register()
                # registration must be a dict with keys:
                # "intent" (str), "execute" (Callable), "description" (str), "examples" (list[str])
                if not isinstance(registration, dict) or "intent" not in registration:
                    print(f"[Skills] '{filename}' register() returned invalid format.", file=sys.stderr)
                    continue

                intent = registration["intent"]
                SkillLoader.SKILL_REGISTRY[intent] = {
                    "execute": registration.get("execute"),
                    "description": registration.get("description", ""),
                    "examples": registration.get("examples", []),
                    "source": filename
                }
                self.loaded.append(skill_name)
                count += 1
                print(f"[Skills] Loaded: '{skill_name}' → intent '{intent}'")
            except Exception as e:
                print(f"[Skills] Failed to load '{filename}': {e}", file=sys.stderr)

        return count

    def inject_into_nlu_prompt(self, prompt_path: str) -> None:
        """
        Append skill intent examples to the NLU prompt file at runtime (in memory only).
        Does NOT write to disk — returns the augmented prompt string.
        Actually: write augmented block to a temp attribute for HestiaNLU to consume.
        Instead, return the skill examples block as a string.
        """
        if not SkillLoader.SKILL_REGISTRY:
            return ""
        lines = ["\n# Dynamically loaded skill intents:\n"]
        for intent, data in SkillLoader.SKILL_REGISTRY.items():
            lines.append(f"# Skill: {intent} — {data['description']}")
            for example in data.get("examples", []):
                lines.append(f'User: {example}')
                lines.append(
                    f'{{"intent": "{intent}", "entities": {{}}, '
                    f'"response": "On it!", "confidence": 0.92}}'
                )
        return "\n".join(lines)

    @classmethod
    def execute_skill(cls, intent: str, entities: dict, memory, raw_query: str = "") -> Optional[str]:
        """
        Execute a registered skill by intent name.
        Passes entities and memory to the skill's execute function.
        Returns the skill's response string, or None if intent not found.
        """
        skill = cls.SKILL_REGISTRY.get(intent)
        if not skill or not skill.get("execute"):
            return None
        try:
            print(f"[SKILL] Executing: {intent}")
            return skill["execute"](entities=entities, memory=memory, raw_query=raw_query)
        except Exception as e:
            print(f"[Skills] Error executing skill '{intent}': {e}", file=sys.stderr)
            return "Something went wrong with that skill."
