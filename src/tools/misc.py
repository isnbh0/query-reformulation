from beir import util

def download_dataset(dataset_name: str, out_dir: str) -> str:
    """데이터셋을 다운로드하고 압축을 해제합니다.
    
    Args:
        dataset_name: 데이터셋 이름
        out_dir: 출력 디렉토리 경로
        
    Returns:
        데이터셋이 저장된 경로
    """
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    return util.download_and_unzip(url, out_dir)