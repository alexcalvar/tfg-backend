import os
import time
from typing import Any, List

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

from src.data.validators import FrameResults, SummaryNode
from src.postprocessing.postprocessing_mode import PostProcessingStrategy
from src.utils.config_loader import ConfigLoader
from src.utils.file_utils import load_json, save_results

from src.utils.logger import get_logger

logger = get_logger(__name__)

class SemanticAnalyzer(PostProcessingStrategy):
    
    def __init__(self, llm_instance : BaseChatModel, user_prompt : str):
        super().__init__()
        self.llm = llm_instance
        self.user_prompt = user_prompt
        
        self.config = ConfigLoader()
        self.system_prompt = self.load_sys_prompts()


    def load_sys_prompts(self) -> str:
        prompts_path = os.path.join(self.config.get_path("config_folder"), "prompts.json")
        config_prompts = load_json(prompts_path)
        categoria_prompts = self.config.get_sys_config("llm_instructions")
        sys_prompt_key = self.config.get_sys_config("semantic_sys_prompt")
        
        base_system_prompt = config_prompts[categoria_prompts][sys_prompt_key]["semantic_summary_prompt"]

        return base_system_prompt



    def execute(self, raw_results: List[FrameResults], results_dir: str) -> SummaryNode:
        base_nodes = self._build_base_nodes(raw_results)
        root = self._build_summary_tree(base_nodes)
        self._save_results(root, results_dir)
        return root



    def _build_base_nodes(self, raw_results: List[FrameResults]) -> List[SummaryNode]:
        """ Convierto los FrameResults extraidos de la respuesta del modelo en los nodos base para 
        la logica de resumenes del arbol de resumenes"""
        interval = self.config.get_video_float("frame_interval")

        nodes = []

        for frame in raw_results:

            node = SummaryNode(
                id=f"L0_{frame.frame_id}",
                nivel=0,
                resumen=frame.descripcion,
                start_frame=frame.frame_id,
                end_frame=frame.frame_id,
                start_timestamp=frame.frame_id * interval,
                end_timestamp=frame.frame_id * interval,
                children=[]
            )

            nodes.append(node)

        return nodes
    


    def _build_summary_tree(self, nodes: List[SummaryNode]) -> SummaryNode:
        """ Recorre y agrupa los nodos hasta que solo quede un unico nodo raiz"""
        batch_size = self.config.get_sys_config_int("descrip_per_batch")
        level = 1

        while len(nodes) > 1:
            nodes = self._reduce_level(nodes, batch_size, level)
            level += 1

        return nodes[0]
    


    def _reduce_level(self, nodes: List[SummaryNode], batch_size: int, level: int) -> List[SummaryNode]:
        """ Procesa una altura horizontal del arbol, agrupando los nodos de esa altura en lotes para luego enviar al 
        lllm"""        
        next_level = []
        indice_lote = 0 

        # se recorre la lista saltando de lote en lote
        for i in range(0, len(nodes), batch_size):
            
            # obtener el grupo de nodos del batch 
            # en el ultimo lote si no se completa, se forma con los nodos restantes
            batch = nodes[i : i + batch_size]

            # enviar el lote al llm
            summary = self._summarize_nodes(batch, level, indice_lote)

            # crear el nodo padre del lote de nodos hijos
            nodo_padre = self._create_parent_node(batch, summary, level, indice_lote)
            
            #almacenar el nodo padre en su altura del arbol correspondiente
            next_level.append(nodo_padre)

            #incrementamos indice para el siguiente
            indice_lote += 1

        return next_level
    
    def _create_parent_node(self,batch: List[SummaryNode],summary: str,level: int,index: int) -> SummaryNode:
        """ Genera el nodo padre de un conjunto de nodos hijos, guardando el resumen global del 
        lote y una lista de los nodos hijos"""
        first = batch[0]
        last = batch[-1]

        return SummaryNode(
            id=f"L{level}_{index}",
            nivel=level,
            resumen=summary,
            start_frame=first.start_frame,
            end_frame=last.end_frame,
            start_timestamp=first.start_timestamp,
            end_timestamp=last.end_timestamp,
            children=batch
        )



    def _summarize_nodes(self, batch: List[SummaryNode], level: int, index: int) -> str:
        """ Coge todos los resumenes del lote y los junta para poder enviarselos al modelo posteriormente"""
        
        #agrupar los resumenes
        text = "\n".join(node.resumen for node in batch)
        node_id = f"L{level}_{index}"
        
        #llamar al llm
        response = self._call_llm(text, node_id)

        return response
    


    def _call_llm(self, text: str, node_id: str) -> str:
        """ Genera el msg para enviar al modelo """
        prompt = self._build_prompt(text)
        messages = [HumanMessage(content=prompt)]

        response =  self._retry_llm_call(messages, node_id)

        return response
    


    def _build_prompt(self, text: str) -> str:
        template = PromptTemplate(
            template=self.system_prompt,
            input_variables=["text_chunk", "user_focus"]
        )

        return template.format(
            text_chunk=text,
            user_focus=self.user_prompt
        )



    def _retry_llm_call(self, messages, node_id: str) -> str:
        """ LLama al modelo y devuelve la respuesta de este """

        max_retries = self.config.get_video_int("max_intents_frame")

        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(messages)
                return response.content

            except Exception as e:
                wait_time = 2 ** attempt
                logger.warning(f"LLM error {node_id} ({attempt+1}/{max_retries}): {e}")
                time.sleep(wait_time)

        logger.error(f"LLM failed for {node_id}")
        return f"[Error en {node_id}]"
    


    def _save_results(self, root: SummaryNode, results_dir: str) -> None:
        path = os.path.join(results_dir, "summary_tree.json")
        save_results(root.model_dump(), path)



