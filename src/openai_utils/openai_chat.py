import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field

from src.tools.logging_tools import LOGGER
from openai.types.completion import Completion

from src.openai_utils.role import Message, MessageRole


def load_secrets(secret_path: str) -> Dict[str, str]:
    """
    Load OpenAI API credentials from a JSON file.

    Args:
        secret_path (str): Path to secrets file relative to home directory.

    Returns:
        Dict[str, str]: Dictionary containing API credentials.
    """
    secrets_file = Path.home() / secret_path
    try:
        with secrets_file.open("r") as file:
            secrets = json.load(file)
            LOGGER.debug(f"Secrets loaded from {secrets_file}")
            return secrets
    except FileNotFoundError:
        LOGGER.error(f"Secrets file not found at {secrets_file}")
        raise
    except json.JSONDecodeError:
        LOGGER.error(f"Invalid JSON format in secrets file at {secrets_file}")
        raise


class ChatConfig(BaseModel):
    """
    Configuration settings for OpenAI chat completions.
    """
    seed: int = Field(default=13, description="Random seed for reproducibility")
    model: str = Field(default="gpt-3.5-turbo", description="OpenAI model to use")
    temperature: float = Field(default=1.0, description="Sampling temperature")
    max_completion_tokens: int = Field(default=1024, description="Maximum tokens in completion")
    secret_path: str = Field(default=".secrets/openai", description="Path to API credentials")
    max_retries: int = Field(default=10, description="Maximum retry attempts")
    logprobs: bool = Field(default=False, description="Whether to return log probabilities")
    response_format: Optional[Dict[str, Any]] = Field(
        default=None, description="Format for completion responses"
    )


class OpenAIChatAPI:
    """
    Client for interacting with the OpenAI chat completions API.
    """

    def __init__(self, config: ChatConfig) -> None:
        """
        Initialize the OpenAI chat API client.

        Args:
            config (ChatConfig): Configuration for the chat client.
        """
        self.config = config
        LOGGER.info(f"Initializing OpenAIChatAPI with config: {self.config}")

        secrets = load_secrets(self.config.secret_path)
        client_kwargs = self._prepare_client_kwargs(secrets)
        self.client = OpenAI(**client_kwargs)
        LOGGER.debug("OpenAI client initialized successfully.")

    def _prepare_client_kwargs(self, secrets: Dict[str, str]) -> Dict[str, str]:
        """
        Prepare keyword arguments for OpenAI client initialization.

        Args:
            secrets (Dict[str, str]): API credentials.

        Returns:
            Dict[str, str]: Keyword arguments for the OpenAI client.
        """
        kwargs = {"api_key": secrets["api_key"]}
        organization = secrets.get("organization")
        if organization:
            kwargs["organization"] = organization
            LOGGER.debug(f"Organization set to: {organization}")
        return kwargs

    def __call__(self, messages: List[Message], **kwargs: Any) -> Completion:
        """
        Generate a chat completion for the given messages.

        Args:
            messages (List[Message]): List of conversation messages.
            **kwargs: Additional arguments for the completion.

        Returns:
            Completion: The chat completion response.
        """
        message_dicts = [message.to_dict() for message in messages]
        # LOGGER.debug(f"Generating completion with messages: {message_dicts}")

        try:
            completion = self.client.with_options(
                max_retries=self.config.max_retries
            ).chat.completions.create(
                seed=self.config.seed,
                messages=message_dicts,
                model=self.config.model,
                temperature=self.config.temperature,
                max_completion_tokens=self.config.max_completion_tokens,
                logprobs=self.config.logprobs,
                response_format=self.config.response_format,
                **kwargs,
            )
            # LOGGER.debug("Completion generated successfully.")
            return completion
        except Exception as e:
            LOGGER.error(f"Error generating completion: {e}")
            raise

    def parse(self, response: Completion) -> Optional[Dict[str, Any]]:
        """
        Parse the completion response.

        Args:
            response (Completion): The completion response to parse.

        Returns:
            Optional[Dict[str, Any]]: Parsed response if format is JSON, else None.
        """
        try:
            content = response.choices[0].message.content
            # LOGGER.debug(f"Parsing response content: {content}")
            if self.config.response_format == {"type": "json_object"}:
                parsed_content = json.loads(content)
                # LOGGER.debug(f"Parsed content: {parsed_content}")
                return parsed_content
            return None
        except (IndexError, KeyError, json.JSONDecodeError) as e:
            LOGGER.error(f"Error parsing response: {e}")
            return None
        

class AsyncOpenAIChatAPI:
    """
    Client for interacting with the OpenAI chat completions API.
    """

    def __init__(self, config: ChatConfig) -> None:
        """
        Initialize the OpenAI chat API client.

        Args:
            config (ChatConfig): Configuration for the chat client.
        """
        self.config = config
        LOGGER.info(f"Initializing OpenAIChatAPI with config: {self.config}")

        secrets = load_secrets(self.config.secret_path)
        client_kwargs = self._prepare_client_kwargs(secrets)
        self.client = AsyncOpenAI(**client_kwargs)
        LOGGER.debug("OpenAI client initialized successfully.")

    def _prepare_client_kwargs(self, secrets: Dict[str, str]) -> Dict[str, str]:
        """
        Prepare keyword arguments for OpenAI client initialization.

        Args:
            secrets (Dict[str, str]): API credentials.

        Returns:
            Dict[str, str]: Keyword arguments for the OpenAI client.
        """
        kwargs = {"api_key": secrets["api_key"]}
        organization = secrets.get("organization")
        if organization:
            kwargs["organization"] = organization
            LOGGER.debug(f"Organization set to: {organization}")
        return kwargs

    async def __call__(self, messages: List[Message], **kwargs: Any) -> Completion:
        """
        Generate a chat completion for the given messages.

        Args:
            messages (List[Message]): List of conversation messages.
            **kwargs: Additional arguments for the completion.

        Returns:
            Completion: The chat completion response.
        """
        message_dicts = [message.to_dict() for message in messages]
        LOGGER.debug(f"Generating completion with messages: {message_dicts}")

        try:
            completion = await self.client.with_options(
                max_retries=self.config.max_retries
            ).chat.completions.create(
                seed=self.config.seed,
                messages=message_dicts,
                model=self.config.model,
                temperature=self.config.temperature,
                max_completion_tokens=self.config.max_completion_tokens,
                logprobs=self.config.logprobs,
                response_format=self.config.response_format,
                **kwargs,
            )
            LOGGER.debug("Completion generated successfully.")
            return completion
        except Exception as e:
            LOGGER.error(f"Error generating completion: {e}")
            raise

    def parse(self, response: Completion) -> Optional[Dict[str, Any]]:
        """
        Parse the completion response.

        Args:
            response (Completion): The completion response to parse.

        Returns:
            Optional[Dict[str, Any]]: Parsed response if format is JSON, else None.
        """
        try:
            content = response.choices[0].message.content
            LOGGER.debug(f"Parsing response content: {content}")
            if self.config.response_format == {"type": "json_object"}:
                parsed_content = json.loads(content)
                LOGGER.debug(f"Parsed content: {parsed_content}")
                return parsed_content
            return None
        except (IndexError, KeyError, json.JSONDecodeError) as e:
            LOGGER.error(f"Error parsing response: {e}")
            return None


if __name__ == "__main__":
    chat_config = ChatConfig(response_format={"type": "json_object"})
    chat_api = OpenAIChatAPI(chat_config)
    product_name = "LG 퓨리케어 오브제 360 공기청정기 알파 AS204NS3A AS204NG3A"
    category = "가전"
    ocr_data = "LG PuriCare Obiet Collection 왜 특별할까요?\\n탈취 성능까지 더 강력해진\\n1등 공기청정기\\n초미세먼지 제거는 기본,\\n냄새 걱정을 덜어주는\\n강력 탈취 청정\\n오염된 공기를 감지해 빠르게 청정\\n공간 청정\\n공간을 빛내는 디자인\\n오브제컬렉션 1115\\n1 본콘텐츠는 저작권법의 보호를 받는 바 무단 전재 복사 배포 등을 금합니다.\\n강력탈취 청정\\n강력 탈취 청정\\n공기 청정은 기본\\n냄새까지 말끔하게\\n강력한 탈취 필터가 거실의 공기 청정은 물론\\n넓게퍼진 냄새까지 청결하게 관리합니다.\\n애연출된 이미지이며, 제품별 색상 및 스펙은 상이합니다\\n360° G 필터\\n탈취 성능까지 갖춘\\n강력한 청정 능력\\n생활 냄새와 극초미세먼지는 물론\\n5대 유해가스를 제거해 온 집안 공기를\\n깨끗하게 바꿔줍니다.\\n360° V 필터 대비 탈취성능 2.5배 강화\\n0.01 um 극초미세먼지 99.999% 제거\\n생활냄새의원인이되는5대유해가스필터링\\n스모그/새집증후군 원인물질 필터링\\n교체형 극세필터로 큰먼지 제거\\n소비자의 이해를 돕기 위해 연출된 이미지입니다.\\n[0.01ml초극세99.999%제거]\\n시험 일시 : \\'22. 11~\\'22.12\\n시험기관:한국건설생활환경시험연구원\\n시험 대상:AS353NGDA모델 기준\\n시험조건:0.01m직경입자의제거효율\\n*시험방법:실험환경 새필터 기준 30ml 첨버, 20분시험환경조건 및 시험 방법\\nRSPACA002-32201.24 Nodifed진형,\\n매콤 매개 조정 및\\n*시험결과로 입자:크기.등 시험 조건과 사용 환경에 따\\n[G 필터 유해가스 누적 제거량 시험 2.5배 증가]\\n*시험기관:글로벌시험인증기관TUV라인란드\\n*시험모델:ADQ75801712(기존V/딸터),4007\\n*시험방법:공기청정기에게존·필터및신규필터를각각장착하여유\\n제거효율이70%이하가될때까지의유해제거량비교\\n\"시험환경:8m(2022),(330±50)°C,55%R\\n클린부스터 1단계 운전\\n*시험결과:기존·필터대비·신규·필터의·유해가스누적거스제\\n(기존필터:3,503mg제거→신규필터:8,993mg제거\\n실험실측정기준으로실사용환경에따라다를 수 있습니다.\\n[유해가스 제거 효율 시험]\\n* 시험 일시 : \\'22. 11\\n*시험 기관 : 한국산업기술시험원(KTL)\\n* 시험 대상 : AS353NGDA\\n*시험조건:온도21±1°C,습도45±5%,시험챔버80-\\n*시험방법+한국공기청정협회\\n*시험결과:암모니어(NH3),초산(대3000H),아세트\\n[스모그 원인물질 제거 효율 시험]\\n* 시험 일시 : \\'20. 12\\n*시험 기관:한국건설생활환경시험연구원\\n* 시험 대상 : AS351NNFA\\n*시험조건:온도21±1°C,습도45±1%시험챔버8.0스\\n*시험방법,\\n준용하여 SO2, NO2 제거율 시험\\n*시험 결과 : CA유해가스 제거 인증 기준 만족\\n※실험실측정기준으로실사용환경에서는달라질수있습니다.\\n한국공기청정협회 미세먼지 센서 인증\\n한국공기청정협회로부터\\n미세먼지 센서 인증을 받았습니다.\\n영국 알레르기 협회 BAF 인증\\n영국알레르기협회로부터\\n알레르기 유발 물질 저감 능력을 인증았습니다.\\n[CA 인증 /CAS 인증]\\n* 인증 기관:한국공기청정협회\\n* 인증 내용:실내공기청정기일반공기청정기/공기청정기용 미세먼지 센서\\n*인증조건:SPSKACA002132실내공기청정기단체표준\\n*인증 유효 기간:\\'22.8~\\'25.8\\n[BAF인증]\\n*인증기관:영국알레르기협회(BAF,BritishAler\\n*인증부문:foreductioninexposuretsm\\n* 인증 유효 기간:~\\'24.4\\nUV팬살균\\n99.99% 세균 제거로\\n믿을 수 있는 위생\\nUVnano로 공기를 내보내는 팬을 99.99% 살균해\\n깨끗한 공기는 물론 공간 내 부유 서균. 99.9%,\\n바이러스 98.4% 제거해 내부 위생정까지 덜어드립니다\\n100%\\n,\\n특화 필터\\n탈부착이가능한특화필터로\\n우리 집 맞춤형 공기 관리\\n내 상황이나 계절에 따라 가장 적합한필터를 선택해 맞춤형\\n으로 우리 집 공기를 관리합니다.\\n펫 특화필터\\n- )\\nG 필터 단독 대비 성능 119% 강화\\n새집 특화필터\\n1\\n필터 단독 대비 성능 131% 강화\\n미래에셋대회에서\\n필터 단독 대비 성능 109% 강화\\n유증기 특화필터\\n오리팜 매몰살하는 유튜기를\\nG 필터 단독 대비 성능 205% 강화\\n알레르겐 특화필터\\n라이프 그레이 홍지표\\n알레르겐 99% 필터링\\n:\\n(\\n=\\n=\\n6000\\n■\\n공간 청정\\n깨끗한 공기를 더 멀리\\n오염된 공기를 더 빨리\\n오염된 공간을 스스로 감지하고\\n클린부스터로 집 안 곳곳 깨끗한 공기를\\n빠르게 보내줍니다.\\n이미지이며, 제품별 색상 및 스펙은 상이합니다.\\n클린부스터\\n깨끗한 공기를\\n최대9m까지\\n멀리 떨어진 공간에도 클린부스터가 빠른속도로\\n최대9m까지 깨끗한 공기를 보내줍니다.\\n* 소비자의 이해를 돕기 위해 연출된 이미지입니다.\\n*자사기준 30r\\'공기청정기모델대비도달거리 및 회전/상 F\\n오토모드\\n오염된 공기를 감지해\\n알아서 청정\\n집 안의 공기 오염도를 실시간 감지해서\\n스스로 청정해요.\\n공간인테리어 가전\\n어떤 공간에도 자연스럽게\\n오브제컬렉션\\n예술적인 소품처럼 공간을 빛내는 디자으로\\n당신의 공간 어디든 자연스럽게 녹아듭니다.\\n샌드 베이지\\n관광객체 등이 있는 데이터는\\n네이처 그린\\n자연의 신고카운을\\n위해연출된 이미지이며, 제품별 색상 및 스펙은 상이합니\\nLG 퓨리케어 360°\\n공기청정기만의 편리함\\nUP 가전\\n지속적으로 업그레이드되는\\n똑똑한 UP 가전\\n사용 중인 제품에 새로운 기능을 추가해\\n매일 더 새롭고 좋아진 제품을 경험할 수 있습니다.\\nUP\\n적금성지 않을수있으나, 즉시사랑을 확인 후 시작하세요.\\n인공지능+\\n효율적인 운전으로\\n에너지 절약\\n공기질이 좋을 때는 내부팬 작동을 멈추고\\n디스플레이 밝기를 낮춰 최저 소비전력으로 운전하고\\n공기질이 나빠지면 팬 작동을 시작하여\\n최대 33% 소비전력을 절약해줍니다.\\n에너지 및\\n*소비자의 이해를 돕기 위해 연출된 이미지이며, 제품별 색상 및 스펙은 상이합니다.\\n[인공지능+]\\nThinQ앱 업그레이드 후 앱의 \\'인공지능모드\\' 또는\\'오토모드\\'에서\\'인공지능+\\'옵션 설정 시운\\n전이 시작됩니다.\\n38.923m21포장기준으로실사용환경에따라다를 수 습니\\n필터 맞춤 업그레이드\\n우리 집에 딱 맞는\\n필터 업그레이드\\n라이프스타일에 맞게 360° G 필터 또는G 펫필터로\\n교체할수있어요.필터교체후디스스플레이에서\\n필터 정보를 설정하여 필터에 맞는\\n공기청정기 기능을 사용해 보세요.\\n360° G 필터 360° G 펫필터\\n비자의이해를 동기 위해 연출된미미지이며,제품별색상및스펙\\n필터 수명 센서\\n교체 시기를 알 수 있는\\n필터 사용량 알림\\n필터 수명 센서로 상 · 하단 필터 시용량을\\n알려주니까 교체시기를 정확하게 알수 있어요.\\n* 시험 결과\\n제품내센서로선출판공과판테스트추적평량비교한결격소최대모차1\\n조도센서\\n주변이 어두워지면\\n알아서 깜깜하게\\n주변이 어두워지면 디스플레이 밝를 낮춰\\n눈부심 없이 편안하게 사용할 수 있습니다.\\n된 이미지이며, 제품별 색상 및 스펙은 상이합니다.\\nPM 1.0 센서\\n눈에 보이지 않는\\n극초미세먼지까지 감지\\n1.0m이하극초미세먼지를 세밀하게\\n감지해 안심할 수 있어요.\\nPM10\\n음성 안내\\n필요한 정보를\\n목소리로 친절하게\\n음성으로 소비 나열 왕위\\n필터 교체 알림\\n청정모드 시작\\n인공지능 기류 청정을 시작합니다.\\n종합청정도 음성 안내\\n현재 종합청정도는 나쁨입니다.\\nLG 공기청정기 필터\\n경기장경기 프랑스 올림픽 패션담배\\nLG 퓨리케어 360° 무빙휠\\n카이유 무역은 이러시 베스트바바라도\\n제품사양\\n기본사양\\n색상 샌드 베이지\\n표준사용면적 (m2) 66.0\\n정격전원 (V / Hz) 220V / 60Hz\\n소비전력 (W)\\n엘리베를 등급\\n디스플레이 타입 4.3\" 터치 LCD\\n외관 디자 인피니티 그릴\\n제품 크기 & 무게\\n자폭증강(10%)\\n347x612x347\\n제품무게(kg) 12.1\\n360° 청정\\n센서\\n취침예약 2/4/8/12 시간\\n리모컨\\n잠금기능\\n미세먼지 농도 표시\\n냄새 표시\\nUP 가전\\n사진제공=연합뉴스] 박성욱\\n청정 표시등\\n청정세기 5단계(자동/약/중/강/터보)\\n싱글청정\\nUV팬살균\\n오토모드\\n청정필터 교체 알림\\n먼지 입자 크기 표시 PM 1.0 / 2.5 / 10\\n클린부스터 세기 5단계(자동/약/중/강/터보)\\n음성안내\\n필터\\n2016.07.07\\n공기청정 필터\\n인증\\nCA 인증\\nCA 센서 인증\\nBAF 인증\\n스마트 기능\\nThinQ(Wi-Fi)\\n마트 진단\\n고지정보\\n품명 및 모델명 AS204NS3A\\n50 5 5 0\\n2024-02\\n제조자 (수입자) LG전자(주)\\n제조국 대한민국\\nKC 인증 필 유무 K\\n800-0008-2699\\nXM070011-22080\\nR-R-LGE-AS204NSDA\\n-\\nR-C-LGE-LCWB-001\\n판매자 LG전자(주)\\nSHANGHAI SHEQU XEPU DAYUEI\\n품질보증기준\\nLG전자서비스센터/1544-7777\\nLG PuriCare Obiet Collection 왜 특별할까요?\\n탈취 성능까지 더 강력해진\\n1등 공기청정기\\n초미세먼지 제거는 기본,\\n냄새 걱정을 덜어주는\\n강력 탈취 청정\\n오염된 공기를 감지해 빠르게 청정\\n공간 청정\\n공간을 빛내는 디자인\\n오브제컬렉션 1020\\n1 본콘텐츠는 저작권법의 보호를 받는 바 무단 전재 복사 배포 등을 금합니다.\\n강력탈취 청정\\n강력 탈취 청정\\n공기 청정은 기본\\n냄새까지 말끔하게\\n강력한 탈취 필터가 거실의 공기 청정은 물론\\n넓게퍼진 냄새까지 청결하게 관리합니다.\\n애연출된 이미지이며, 제품별 색상 및 스펙은 상이합니다\\n360° G 필터\\n탈취 성능까지 갖춘\\n강력한 청정 능력\\n생활 냄새와 극초미세먼지는 물론\\n5대 유해가스를 제거해 온 집안 공기를\\n깨끗하게 바꿔줍니다.\\n360° V 필터 대비 탈취성능 2.5배 강화\\n0.01 um 극초미세먼지 99.999% 제거\\n생활냄새의원인이되는5대유해가스필터링\\n스모그/새집증후군 원인물질 필터링\\n교체형 극세필터로 큰먼지 제거\\n소비자의 이해를 돕기 위해 연출된 이미지입니다.\\n[0.01ml초극세99.999%제거]\\n시험 일시 : \\'22. 11~\\'22.12\\n시험기관:한국건설생활환경시험연구원\\n시험 대상:AS353NGDA모델 기준\\n시험조건:0.01m직경입자의제거효율\\n*시험방법:실험환경 새필터 기준 30ml 첨버, 20분시험환경조건 및 시험 방법\\nRSPACA002-32201.24 Nodifed진형,\\n매콤 매개 조정 및\\n*시험결과로 입자:크기.등 시험 조건과 사용 환경에 따\\n[G 필터 유해가스 누적 제거량 시험 2.5배 증가]\\n*시험기관:글로벌시험인증기관TUV라인란드\\n*시험모델:ADQ75801712(기존V/딸터),4007\\n*시험방법:공기청정기에게존·필터및신규필터를각각장착하여유\\n제거효율이70%이하가될때까지의유해제거량비교\\n\"시험환경:8m(2022),(330±50)°C,55%R\\n클린부스터 1단계 운전\\n*시험결과:기존·필터대비·신규·필터의·유해가스누적거스제\\n(기존필터:3,503mg제거→신규필터:8,993mg제거\\n실험실측정기준으로실사용환경에따라다를 수 있습니다.\\n[유해가스 제거 효율 시험]\\n* 시험 일시 : \\'22. 11\\n*시험 기관 : 한국산업기술시험원(KTL)\\n* 시험 대상 : AS353NGDA\\n*시험조건:온도21±1°C,습도45±5%,시험챔버80-\\n*시험방법+한국공기청정협회\\n*시험결과:암모니어(NH3),초산(대3000H),아세트\\n[스모그 원인물질 제거 효율 시험]\\n* 시험 일시 : \\'20. 12\\n*시험 기관:한국건설생활환경시험연구원\\n* 시험 대상 : AS351NNFA\\n*시험조건:온도21±1°C,습도45±1%시험챔버8.0스\\n*시험방법,\\n준용하여 SO2, NO2 제거율 시험\\n*시험 결과 : CA유해가스 제거 인증 기준 만족\\n※실험실측정기준으로실사용환경에서는달라질수있습니다.\\n한국공기청정협회 미세먼지 센서 인증\\n한국공기청정협회로부터\\n미세먼지 센서 인증을 받았습니다.\\n영국 알레르기 협회 BAF 인증\\n영국알레르기협회로부터\\n알레르기 유발 물질 저감 능력을 인증았습니다.\\n[CA 인증 /CAS 인증]\\n* 인증 기관:한국공기청정협회\\n* 인증 내용:실내공기청정기일반공기청정기/공기청정기용 미세먼지 센서\\n*인증조건:SPSKACA002132실내공기청정기단체표준\\n*인증 유효 기간:\\'22.8~\\'25.8\\n[BAF인증]\\n*인증기관:영국알레르기협회(BAF,BritishAler\\n*인증부문:foreductioninexposuretsm\\n* 인증 유효 기간:~\\'24.4\\nUV팬살균\\n99.99% 세균 제거로\\n믿을 수 있는 위생\\nUVnano로 공기를 내보내는 팬을 99.99% 살균해\\n깨끗한 공기는 물론 공간 내 부유 서균. 99.9%,\\n바이러스 98.4% 제거해 내부 위생정까지 덜어드립니다\\n100%\\n,\\n특화 필터\\n탈부착이가능한특화필터로\\n우리 집 맞춤형 공기 관리\\n내 상황이나 계절에 따라 가장 적합한필터를 선택해 맞춤형\\n으로 우리 집 공기를 관리합니다.\\n펫 특화필터\\n- )\\nG 필터 단독 대비 성능 119% 강화\\n새집 특화필터\\n1\\n필터 단독 대비 성능 131% 강화\\n미래에셋대회에서\\n필터 단독 대비 성능 109% 강화\\n유증기 특화필터\\n오리팜 매몰살하는 유튜기를\\nG 필터 단독 대비 성능 205% 강화\\n알레르겐 특화필터\\n라이프 그레이 홍지표\\n알레르겐 99% 필터링\\n:\\n(\\n=\\n=\\n6000\\n■\\n공간 청정\\n깨끗한 공기를 더 멀리\\n오염된 공기를 더 빨리\\n오염된 공간을 스스로 감지하고\\n클린부스터로 집 안 곳곳 깨끗한 공기를\\n빠르게 보내줍니다.\\n이미지이며, 제품별 색상 및 스펙은 상이합니다.\\n클린부스터\\n깨끗한 공기를\\n최대9m까지\\n멀리 떨어진 공간에도 클린부스터가 빠른속도로\\n최대9m까지 깨끗한 공기를 보내줍니다.\\n* 소비자의 이해를 돕기 위해 연출된 이미지입니다.\\n*자사기준 30r\\'공기청정기모델대비도달거리 및 회전/상 F\\n오토모드\\n오염된 공기를 감지해\\n알아서 청정\\n집 안의 공기 오염도를 실시간 감지해서\\n스스로 청정해요.\\n공간인테리어 가전\\n어떤 공간에도 자연스럽게\\n오브제컬렉션\\n예술적인 소품처럼 공간을 빛내는 디자으로\\n당신의 공간 어디든 자연스럽게 녹아듭니다.\\n샌드 베이지\\n관광객체 등이 있는 데이터는\\n네이처 그린\\n자연의 신고카운을\\n위해연출된 이미지이며, 제품별 색상 및 스펙은 상이합니\\nLG 퓨리케어 360°\\n공기청정기만의 편리함\\nUP 가전\\n지속적으로 업그레이드되는\\n똑똑한 UP 가전\\n사용 중인 제품에 새로운 기능을 추가해\\n매일 더 새롭고 좋아진 제품을 경험할 수 있습니다.\\nUP\\n적금성지 않을수있으나, 즉시사랑을 확인 후 시작하세요.\\n인공지능+\\n효율적인 운전으로\\n에너지 절약\\n공기질이 좋을 때는 내부팬 작동을 멈추고\\n디스플레이 밝기를 낮춰 최저 소비전력으로 운전하고\\n공기질이 나빠지면 팬 작동을 시작하여\\n최대 33% 소비전력을 절약해줍니다.\\n에너지 및\\n*소비자의 이해를 돕기 위해 연출된 이미지이며, 제품별 색상 및 스펙은 상이합니다.\\n[인공지능+]\\nThinQ앱 업그레이드 후 앱의 \\'인공지능모드\\' 또는\\'오토모드\\'에서\\'인공지능+\\'옵션 설정 시운\\n전이 시작됩니다.\\n38.923m21포장기준으로실사용환경에따라다를 수 습니\\n필터 맞춤 업그레이드\\n우리 집에 딱 맞는\\n필터 업그레이드\\n라이프스타일에 맞게 360° G 필터 또는G 펫필터로\\n교체할수있어요.필터교체후디스스플레이에서\\n필터 정보를 설정하여 필터에 맞는\\n공기청정기 기능을 사용해 보세요.\\n360° G 필터 360° G 펫필터\\n비자의이해를 동기 위해 연출된미미지이며,제품별색상및스펙\\n필터 수명 센서\\n교체 시기를 알 수 있는\\n필터 사용량 알림\\n필터 수명 센서로 상 · 하단 필터 시용량을\\n알려주니까 교체시기를 정확하게 알수 있어요.\\n* 시험 결과\\n제품내센서로선출판공과판테스트추적평량비교한결격소최대모차1\\n조도센서\\n주변이 어두워지면\\n알아서 깜깜하게\\n주변이 어두워지면 디스플레이 밝를 낮춰\\n눈부심 없이 편안하게 사용할 수 있습니다.\\n된 이미지이며, 제품별 색상 및 스펙은 상이합니다.\\nPM 1.0 센서\\n눈에 보이지 않는\\n극초미세먼지까지 감지\\n1.0m이하극초미세먼지를 세밀하게\\n감지해 안심할 수 있어요.\\nPM10\\n음성 안내\\n필요한 정보를\\n목소리로 친절하게\\n음성으로 소비 나열 왕위\\n필터 교체 알림\\n청정모드 시작\\n인공지능 기류 청정을 시작합니다.\\n종합청정도 음성 안내\\n현재 종합청정도는 나쁨입니다.\\nLG 공기청정기 필터\\n경기장경기 프랑스 올림픽 패션담배\\nLG 퓨리케어 360° 무빙휠\\n카이유 무역은 이러시 베스트바바라도\\n지구사업\\n기본사양\\n색상 네이처 그린\\n표준사용면적 (m2) 66.0\\n정격전원 (V / Hz) 220V / 60Hz\\n소비전력 (W)\\n엘리베를 등급\\n디스플레이 타입 4.3\" 터치 LCD\\n외관 디자 인피니티 그릴\\n제품 크기 & 무게\\n자폭증강(10%)\\n347x612x347\\n제품무게(kg) 12.1\\n360° 청정\\n센서\\n취침예약 2/4/8/12 시간\\n리모컨\\n잠금기능\\n미세먼지 농도 표시\\n냄새 표시\\nUP 가전\\n사진제공=연합뉴스] 박성욱\\n청정 표시등\\n청정세기 5단계(자동/약/중/강/터보)\\n싱글청정\\nUV팬살균\\n청정필터 교체 알림\\n먼지 입자 크기 표시 PM 1.0 / 2.5 / 10\\n클린부스터 세기 5단계(자동/약/중/강/터보)\\n음성안내\\n필터\\n2016.07.07\\n공기청정 필터\\n인증\\nCA 인증\\nCA 센서 인증\\nBAF 인증\\n스마트 기능\\nThinQ(Wi-Fi)\\n마트 진단\\n고지정보\\n품명 및 모델명 AS204NG3A\\n50 5 5 0\\n2024-02\\n제조자 (수입자) LG전자(주)\\n제조국 대한민국\\nKC 인증 필 유무 K\\n800-0008-2699\\nXM070011-22080\\nR-R-LGE-AS204NSDA\\n-\\nR-C-LGE-LCWB-001\\n판매자 LG전자(주)\\nSHANGHAI SHEQU XEPU DAYUEI\\n품질보증기준\\nLG전자서비스센터/1544-7777\""
    
    # response = chat_api(
    #     [
    #         Message(role=MessageRole.SYSTEM, content=OCR_SYSTEM_PROMPT),
    #         Message(role=MessageRole.USER, content=OCR_USER_PROMPT.format(product_name, category, ocr_data)),
    #     ]
    # )
    # LOGGER.info(chat_api.parse(response)['filtered_data'])