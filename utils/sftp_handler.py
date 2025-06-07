import os
import paramiko
from stat import S_ISDIR
from typing import Optional, Tuple, Callable

# Carrega as credenciais das variáveis de ambiente configuradas
# (no seu .env localmente, ou no dashboard do Render)
HG_HOST = os.getenv("HG_HOST")
HG_USER = os.getenv("HG_USER")
HG_PASS = os.getenv("HG_PASS")
HG_PORT = int(os.getenv("HG_PORT", 22))

def sftp_connect() -> Optional[Tuple[paramiko.SFTPClient, paramiko.Transport]]:
    """Cria e retorna um cliente SFTP conectado."""
    if not all([HG_HOST, HG_USER, HG_PASS]):
        print("[SFTP ERRO] Variáveis de ambiente (HG_HOST, HG_USER, HG_PASS) não estão configuradas.")
        return None, None
    try:
        transport = paramiko.Transport((HG_HOST, HG_PORT))
        transport.connect(username=HG_USER, password=HG_PASS)
        sftp = paramiko.SFTPClient.from_transport(transport)
        print(f"[SFTP] Conexão com {HG_HOST} bem-sucedida.")
        return sftp, transport
    except Exception as e:
        print(f"[SFTP ERRO] Falha ao conectar: {e}")
        return None, None

def _ensure_remote_dir_exists(sftp: paramiko.SFTPClient, remote_path: str):
    """
    Função auxiliar interna para garantir que um diretório remoto exista,
    criando-o recursivamente se necessário.
    """
    remote_dir = os.path.dirname(remote_path.replace("\\", "/"))
    if not remote_dir or remote_dir == '.':
        return # Não precisa criar diretório se o caminho for na raiz
        
    # Divide o caminho em partes para criar cada nível do diretório
    dirs = remote_dir.split('/')
    current_dir = ''
    for d in dirs:
        if not d: continue # Ignora strings vazias (ex: de um caminho começando com /)
        # Para caminhos relativos como 'public_html/...', current_dir começa sem a barra
        if current_dir:
            current_dir += '/' + d
        else:
            current_dir = d
        
        try:
            # Verifica se o diretório/arquivo existe
            sftp.stat(current_dir)
        except FileNotFoundError:
            # Se não existe, cria o diretório
            print(f"[SFTP] Criando diretório remoto: {current_dir}")
            sftp.mkdir(current_dir)

def upload_file_sftp(local_path: str, remote_path: str, progress_callback: Optional[Callable[[int, int], None]] = None) -> bool:
    """Faz upload de um arquivo local para um caminho remoto via SFTP, com callback de progresso."""
    sftp, transport = sftp_connect()
    if not sftp: return False
    
    try:
        _ensure_remote_dir_exists(sftp, remote_path)
        
        print(f"[SFTP] Fazendo upload de '{local_path}' para '{remote_path}'...")
        # Passa a função de callback para o método .put() do paramiko
        sftp.put(local_path, remote_path.replace("\\", "/"), callback=progress_callback)
        print(f"[SFTP] Upload de '{os.path.basename(local_path)}' concluído.")
        return True
    except Exception as e:
        print(f"[SFTP ERRO] Falha no upload: {e}"); return False
    finally:
        if sftp: sftp.close()
        if transport: transport.close()

def download_file_sftp(remote_path: str, local_path: str, progress_callback: Optional[Callable[[int, int], None]] = None) -> bool:
    """Baixa um arquivo de um caminho remoto para um local via SFTP, com callback de progresso."""
    sftp, transport = sftp_connect()
    if not sftp: return False
    
    try:
        print(f"[SFTP] Baixando de '{remote_path}' para '{local_path}'...")
        # Passa a função de callback para o método .get() do paramiko
        sftp.get(remote_path.replace("\\", "/"), local_path, callback=progress_callback)
        print(f"[SFTP] Download de '{os.path.basename(remote_path)}' concluído.")
        return True
    except Exception as e:
        print(f"[SFTP ERRO] Falha no download: {e}"); return False
    finally:
        if sftp: sftp.close()
        if transport: transport.close()

def delete_file_sftp(remote_path: str) -> bool:
    """Deleta um arquivo em um caminho remoto via SFTP."""
    sftp, transport = sftp_connect()
    if not sftp: return False
    
    try:
        print(f"[SFTP] Deletando arquivo remoto: '{remote_path}'...")
        sftp.remove(remote_path.replace("\\", "/"))
        print(f"[SFTP] Arquivo '{os.path.basename(remote_path)}' deletado.")
        return True
    except FileNotFoundError:
        print(f"[SFTP AVISO] Tentativa de deletar arquivo que não existe: {remote_path}")
        return True # Considera sucesso se o arquivo já não existe
    except Exception as e:
        print(f"[SFTP ERRO] Falha ao deletar '{remote_path}': {e}"); return False
    finally:
        if sftp: sftp.close()
        if transport: transport.close()