import logging
import sys
import os

def get_logger(name: str):
    """
    Configura y devuelve un logger que  
    imprime en consola (INFO+) y guarda en archivo (DEBUG+).
    """
    logger = logging.getLogger(name)
    
    # solo configurar el logger si no tiene handlers ya asignados 
    if not logger.handlers:
        # decirle al logger que acepte tods los mensajes
        logger.setLevel(logging.DEBUG) 

        # formato: [FECHA-HORA] [NIVEL] [ARCHIVO] - Mensaje
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # handler para la consola 
        console_handler = logging.StreamHandler(sys.stdout)

        # consola solo enviamos de info para arriba 
        console_handler.setLevel(logging.INFO) 
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        #  handler para el archivo (historial en disco)
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(os.path.join(log_dir, "app.log"), encoding="utf-8")
        
        # Al archivo enviar todo (DEBUG para arriba)
        file_handler.setLevel(logging.DEBUG) 
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger