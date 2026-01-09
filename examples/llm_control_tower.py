#!/usr/bin/env python3
"""
LLM Control Tower - Conversational Pipeline Management
======================================================

Interface de chat pour contrôler LifeCore en langage naturel.
Utilise Gemini ou OpenAI pour interpreter les commandes.

Run:
    python examples/llm_control_tower.py
    
Environment:
    GOOGLE_API_KEY=xxx  # Pour Gemini
    OPENAI_API_KEY=xxx  # Pour OpenAI (optionnel)
"""

import os
import sys
import json
import re
from typing import Optional, List, Dict, Tuple

sys.path.insert(0, '.')

try:
    import gradio as gr
except ImportError:
    print("Gradio not installed. Run: pip install gradio")
    sys.exit(1)

from lifecore.llm_interface import LLMDispatcher, SYSTEM_PROMPT
from lifecore.pipeline import Pipeline

# Try to import LLM libraries
GEMINI_AVAILABLE = False
OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    pass

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    pass


# === LLM INTEGRATION ===

class LLMChat:
    """Chat avec LLM pour contrôler le pipeline."""
    
    def __init__(self):
        self.dispatcher = LLMDispatcher()
        self.conversation_history: List[Dict] = []
        self.llm_provider = None
        self.model = None
        self._setup_llm()
    
    def _setup_llm(self):
        """Configure le LLM disponible."""
        if GEMINI_AVAILABLE and os.environ.get("GOOGLE_API_KEY"):
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.llm_provider = "gemini"
            print("✅ Using Gemini")
        elif OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
            openai.api_key = os.environ["OPENAI_API_KEY"]
            self.llm_provider = "openai"
            print("✅ Using OpenAI")
        else:
            self.llm_provider = "local"
            print("⚠️ No LLM API key found. Using local rule-based parsing.")
    
    def _parse_local(self, message: str) -> Optional[Tuple[str, Dict]]:
        """Parse local sans LLM - règles simples."""
        message_lower = message.lower()
        
        # Patterns de création
        if any(kw in message_lower for kw in ["crée", "créer", "create", "nouveau", "new"]):
            if "pipeline" in message_lower or "entrepôt" in message_lower or "warehouse" in message_lower:
                # Extraire les nombres
                robots = 50
                stations = 20
                docks = 10
                
                numbers = re.findall(r'\d+', message)
                if len(numbers) >= 1:
                    robots = int(numbers[0])
                if len(numbers) >= 2:
                    stations = int(numbers[1])
                if len(numbers) >= 3:
                    docks = int(numbers[2])
                
                ptype = "manufacturing" if "manufacturing" in message_lower or "fabrication" in message_lower else "fulfillment"
                
                return ("create_pipeline", {
                    "pipeline_type": ptype,
                    "robots": robots,
                    "stations": stations,
                    "docks": docks
                })
        
        # Ajout de jobs
        if any(kw in message_lower for kw in ["ajoute", "add", "commande", "job", "order"]):
            count = 10
            priority = 5
            deadline = 60
            
            numbers = re.findall(r'\d+', message)
            if numbers:
                count = int(numbers[0])
            
            if "prime" in message_lower or "urgent" in message_lower:
                priority = 9
                deadline = 50
            elif "economy" in message_lower or "économique" in message_lower:
                priority = 2
                deadline = 200
            
            return ("add_jobs", {
                "count": count,
                "priority": priority,
                "deadline": deadline
            })
        
        # Simulation
        if any(kw in message_lower for kw in ["simule", "simulate", "run", "lance", "execute"]):
            steps = 100
            numbers = re.findall(r'\d+', message)
            if numbers:
                steps = int(numbers[0])
            return ("run_simulation", {"steps": steps})
        
        # Status
        if any(kw in message_lower for kw in ["status", "état", "stats", "statistiques", "résultat"]):
            return ("get_status", {})
        
        # Analyse
        if any(kw in message_lower for kw in ["analyse", "bottleneck", "goulot", "problème", "pourquoi"]):
            return ("analyze_bottlenecks", {})
        
        # Scénario
        if any(kw in message_lower for kw in ["scénario", "scenario", "what if", "si on"]):
            if "ressource" in message_lower or "robot" in message_lower:
                return ("simulate_scenario", {
                    "scenario_type": "add_resources",
                    "parameters": {"resources": {"robot": 10}}
                })
            else:
                return ("simulate_scenario", {
                    "scenario_type": "demand_spike",
                    "parameters": {"count": 30}
                })
        
        # Gantt
        if any(kw in message_lower for kw in ["gantt", "export", "diagramme"]):
            return ("export_gantt", {})
        
        return None
    
    def _call_gemini(self, message: str) -> Optional[Tuple[str, Dict]]:
        """Appelle Gemini pour parser la commande."""
        tools_schema = self.dispatcher.get_tools_schema()
        
        prompt = f"""Tu es un assistant pour contrôler un pipeline de production.
        
Voici les tools disponibles:
{json.dumps(tools_schema, indent=2)}

L'utilisateur dit: "{message}"

Réponds avec un JSON contenant:
- "tool": nom du tool à appeler (ou null si c'est juste une question)
- "parameters": paramètres du tool
- "response": ta réponse à l'utilisateur

Exemple:
{{"tool": "create_pipeline", "parameters": {{"pipeline_type": "fulfillment", "robots": 50}}, "response": "Je crée un pipeline fulfillment avec 50 robots..."}}
"""

        try:
            response = self.model.generate_content(prompt)
            text = response.text
            
            # Extraire le JSON
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if data.get("tool"):
                    return (data["tool"], data.get("parameters", {}))
        except Exception as e:
            print(f"Gemini error: {e}")
        
        return None
    
    def chat(self, message: str) -> Tuple[str, str, str]:
        """
        Traite un message utilisateur.
        
        Returns:
            (response_text, gantt_svg, stats_html)
        """
        # Parser le message
        tool_call = None
        
        if self.llm_provider == "gemini":
            tool_call = self._call_gemini(message)
        
        if not tool_call:
            tool_call = self._parse_local(message)
        
        # Exécuter le tool
        if tool_call:
            tool_name, params = tool_call
            result = self.dispatcher.execute_tool(tool_name, params)
            
            # Construire la réponse
            if result.success:
                response = f"✅ **{tool_name}**: {result.message}\n\n"
                if result.data:
                    # Formater les données importantes
                    if "jobs_completed" in result.data:
                        response += f"📊 Completed: {result.data['jobs_completed']}/{result.data['jobs_created']}\n"
                    if "bottlenecks" in result.data:
                        for bn in result.data["bottlenecks"]:
                            response += f"⚠️ {bn['type']}: {bn.get('resource', '')} ({bn.get('severity', '')})\n"
                    if "recommendations" in result.data:
                        response += "\n**Recommandations:**\n"
                        for rec in result.data["recommendations"]:
                            response += f"• {rec}\n"
            else:
                response = f"❌ **Erreur**: {result.message}"
        else:
            response = "🤔 Je n'ai pas compris. Essayez:\n"
            response += "• 'Crée un pipeline avec 50 robots'\n"
            response += "• 'Ajoute 100 commandes Prime'\n"
            response += "• 'Simule 200 steps'\n"
            response += "• 'Analyse les bottlenecks'\n"
        
        # Obtenir le Gantt et les stats
        gantt_svg = self._get_gantt_svg()
        stats_html = self._get_stats_html()
        
        return response, gantt_svg, stats_html
    
    def _get_gantt_svg(self) -> str:
        """Génère le SVG du Gantt."""
        from examples.pipeline_visualizer import render_gantt_svg
        return render_gantt_svg(self.dispatcher.controller.pipeline)
    
    def _get_stats_html(self) -> str:
        """Génère le HTML des stats."""
        from examples.pipeline_visualizer import render_stats_html
        if self.dispatcher.controller.pipeline:
            return render_stats_html(self.dispatcher.controller.pipeline)
        return "<div style='color:#888; padding:20px;'>Pas encore de pipeline</div>"


# === GRADIO INTERFACE ===

def create_chat_interface():
    """Crée l'interface Gradio."""
    chat = LLMChat()
    
    with gr.Blocks(title="LLM Control Tower") as demo:
        gr.Markdown("""
        # 🎛️ LLM Control Tower
        **Contrôlez votre pipeline en langage naturel**
        
        Exemples de commandes:
        - "Crée un entrepôt avec 50 robots et 20 stations"
        - "Ajoute 100 commandes Prime urgentes"
        - "Simule 200 steps"
        - "Pourquoi le taux de livraison est bas?"
        - "Que se passe-t-il si on ajoute 20 robots?"
        """)
        
        with gr.Row():
            # Left: Chat
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=400,
                    type="messages"
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Tapez votre commande...",
                        label="Message",
                        scale=4
                    )
                    send_btn = gr.Button("Envoyer", variant="primary", scale=1)
                
                # Quick actions
                gr.Markdown("### Actions rapides")
                with gr.Row():
                    quick_create = gr.Button("🏭 Créer Pipeline")
                    quick_jobs = gr.Button("📦 +50 Jobs")
                    quick_sim = gr.Button("▶️ Simuler")
                    quick_analyze = gr.Button("🔍 Analyser")
            
            # Right: Visualization
            with gr.Column(scale=1):
                gantt_output = gr.HTML(label="Gantt Chart")
                stats_output = gr.HTML(label="Statistics")
        
        # State
        history = gr.State([])
        
        def respond(message, chat_history):
            response, gantt, stats = chat.chat(message)
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": response})
            return "", chat_history, gantt, stats
        
        def quick_action(action, chat_history):
            actions = {
                "create": "Crée un pipeline fulfillment avec 50 robots et 20 stations",
                "jobs": "Ajoute 50 commandes avec priorité normale",
                "simulate": "Simule 100 steps",
                "analyze": "Analyse les bottlenecks"
            }
            message = actions.get(action, "")
            return respond(message, chat_history)
        
        # Wire up
        send_btn.click(
            respond,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot, gantt_output, stats_output]
        )
        
        msg_input.submit(
            respond,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot, gantt_output, stats_output]
        )
        
        quick_create.click(
            lambda h: quick_action("create", h),
            inputs=[chatbot],
            outputs=[msg_input, chatbot, gantt_output, stats_output]
        )
        
        quick_jobs.click(
            lambda h: quick_action("jobs", h),
            inputs=[chatbot],
            outputs=[msg_input, chatbot, gantt_output, stats_output]
        )
        
        quick_sim.click(
            lambda h: quick_action("simulate", h),
            inputs=[chatbot],
            outputs=[msg_input, chatbot, gantt_output, stats_output]
        )
        
        quick_analyze.click(
            lambda h: quick_action("analyze", h),
            inputs=[chatbot],
            outputs=[msg_input, chatbot, gantt_output, stats_output]
        )
    
    return demo


if __name__ == "__main__":
    print("🎛️ LLM Control Tower")
    print("=" * 50)
    
    if not GEMINI_AVAILABLE and not OPENAI_AVAILABLE:
        print("⚠️ No LLM library found. Install with:")
        print("   pip install google-generativeai  # For Gemini")
        print("   pip install openai               # For OpenAI")
        print()
        print("ℹ️ Running with local rule-based parsing...")
    
    print()
    print("Starting server...")
    
    demo = create_chat_interface()
    demo.launch(share=False)
