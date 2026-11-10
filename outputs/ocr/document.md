# Table 1 — Population of the United States and of its Outlying Possessions (1910)

- 출처: U.S. Bureau of the Census (1913), *Thirteenth Census of the United States Taken in the Year 1910, Volume I: Population — Table 1*
- 원문: https://www2.census.gov/library/publications/decennial/1910/volume-1/volume-1-p3.pdf — PDF 2쪽 / 인쇄 22쪽
- 입력 이미지: `input/census-table.png` (PDF 좌표 [235, 283, 625, 463] pt 영역, 200 dpi) · 공유 조건: 미국 연방정부 발행물 — 퍼블릭 도메인
- 각 값의 원본 위치(픽셀 bbox)는 `cells.csv`, 시각 확인은 `review_overlay.png` 참고

## 복원한 표

| AREA | Gross area (land and water) in square miles | Population: 1910 | 검토 |
|---|---:|---:|---|
| **United States, with outlying possessions** | **3,743,306** | **101,115,487** | 확인 |
| United States, exclusive of outlying possessions | 3,026,789 | 91,972,266 | 확인 |
| Outlying possessions | 716,517 | 9,143,221 | 확인 |
| &nbsp;&nbsp;&nbsp;&nbsp;Alaska | 590,884 | 64,356 | 확인 |
| &nbsp;&nbsp;&nbsp;&nbsp;Hawaii | 6,449 | 191,909 | 확인 |
| &nbsp;&nbsp;&nbsp;&nbsp;Porto Rico | 3,435 | 1,118,012 | 확인 |
| &nbsp;&nbsp;&nbsp;&nbsp;Military and naval service abroad | — | 55,608 | 확인 |
| &nbsp;&nbsp;&nbsp;&nbsp;Philippine Islands | 115,026 | 7,635,426 <sup>1</sup> | 수정 |
| &nbsp;&nbsp;&nbsp;&nbsp;Panama Canal Zone | 436 | 62,810 <sup>2</sup> | 수정 |
| &nbsp;&nbsp;&nbsp;&nbsp;Guam (population estimated) | 210 | 9,000 | 확인 |
| &nbsp;&nbsp;&nbsp;&nbsp;Samoa (population estimated) | 77 | 6,100 | 확인 |

<sup>1</sup> Population in 1903. &nbsp; <sup>2</sup> Population in 1912. (원본 각주)

— : 원본에 점선만 있고 값이 없는 칸(0이 아님).

## 검증 1 — 합계 산술 검사

| 검사 | OCR 원시값 | 결과 | 사람 검토 후 | 결과 |
|---|---|---|---|---|
| 면적: 전체 = 본토 + 해외영토 | 3,743,306 vs 3,743,306 | ✅ | 3,743,306 vs 3,743,306 | ✅ |
| 면적: 해외영토 = 하위 지역 합 | 716,517 vs 716,517 | ✅ | 716,517 vs 716,517 | ✅ |
| 인구: 전체 = 본토 + 해외영토 | 101,115,487 vs 101,115,487 | ✅ | 101,115,487 vs 101,115,487 | ✅ |
| 인구: 해외영토 = 하위 지역 합 | 9,143,221 vs 19,343,221 | ❌ | 9,143,221 vs 9,143,221 | ✅ |

## 검증 2 — 자동 '확인 필요' 표시 (사람 검토 전)

규칙: OCR 점수 < 0.9, 숫자 칸의 숫자 외 문자, 합계 검사 불일치 열

| 위치 | 사유 |
|---|---|
| Hawaii / 면적 | OCR 점수 0.808 < 0.9 |
| 제목 영역 OCR#0 | 제목 OCR 점수 0.523: 'I OIQBE' |
| Outlying possessions / 인구 | 합계 검사 불일치(인구: 해외영토 = 하위 지역 합: 9,143,221 ≠ 19,343,221) — 이 열 전체 재확인 |
| Alaska / 인구 | 합계 검사 불일치(인구: 해외영토 = 하위 지역 합: 9,143,221 ≠ 19,343,221) — 이 열 전체 재확인 |
| Hawaii / 인구 | 합계 검사 불일치(인구: 해외영토 = 하위 지역 합: 9,143,221 ≠ 19,343,221) — 이 열 전체 재확인 |
| Porto Rico / 인구 | 합계 검사 불일치(인구: 해외영토 = 하위 지역 합: 9,143,221 ≠ 19,343,221) — 이 열 전체 재확인 |
| Military and naval service abroad / 인구 | 합계 검사 불일치(인구: 해외영토 = 하위 지역 합: 9,143,221 ≠ 19,343,221) — 이 열 전체 재확인 |
| Philippine Islands / 인구 | 합계 검사 불일치(인구: 해외영토 = 하위 지역 합: 9,143,221 ≠ 19,343,221) — 이 열 전체 재확인 |
| Panama Canal Zone / 인구 | 합계 검사 불일치(인구: 해외영토 = 하위 지역 합: 9,143,221 ≠ 19,343,221) — 이 열 전체 재확인 |
| Guam (population estimated) / 인구 | 합계 검사 불일치(인구: 해외영토 = 하위 지역 합: 9,143,221 ≠ 19,343,221) — 이 열 전체 재확인 |
| Samoa (population estimated) / 인구 | 합계 검사 불일치(인구: 해외영토 = 하위 지역 합: 9,143,221 ≠ 19,343,221) — 이 열 전체 재확인 |

## 수정 내역 (원시 인식 → 원본 대조 후)

| OCR# | 위치(행/열) | 원시 인식 | 수정값 | 원본 bbox(px) | 근거 |
|---|---|---|---|---|---|
| 0 | 표 제목(왼쪽 위) | `I OIQBE` (0.523) | `Table 1` | [50, 25, 142, 48] | 굵은 장식체 'Table 1'을 잘못 인식(점수 0.523). 원본 확대 시 'Table 1'이 분명함. 데이터 값이 아니므로 표 내용에는 영향 없음 |
| 29 | Philippine Islands / Population | `17, 635, 426` (0.996) | `7, 635, 426` | [951, 342, 1068, 370] | 숫자 앞의 위첨자 각주 기호 ¹ 이 숫자에 붙어 '17,635,426'으로 읽힘(점수 0.996으로 높았음). 원본 확대 시 작은 위첨자 1 + 7,635,426. 인구 합계 검사(해외영토 9,143,221)도 이 값일 때만 일치 |
| 32 | Panama Canal Zone / Population | `262, 810` (0.949) | `62, 810` | [981, 362, 1070, 393] | 위첨자 각주 기호 ² 가 숫자에 붙어 '262,810'으로 읽힘(점수 0.949). 원본 확대 시 위첨자 2 + 62,810. 수정 후 인구 합계 검사 일치 |

확대 이미지: `review_evidence/ocr<번호>.png`

## 확인 필요로 남긴 항목

- 없음 (위 자동 표시 항목은 모두 원본 확대 대조 + 합계 검사로 해소)

## 처리 범위와 한계

- 처리 범위: Census 1910 Volume I 인쇄 22쪽의 Table 1 한 개(표 영역만 잘라낸 이미지). 같은 쪽 하단의 세계 비교표와 본문 텍스트, FRASER No. 11 표는 처리하지 않았다.
- 모든 숫자 칸은 원본 확대 대조 + 합계 산술 검사(4개 관계)로 확인했다. 산술 검사는 '서로 상쇄되는 두 오류'는 잡지 못하므로 칸 단위 육안 대조를 함께 했다.
- Military and naval service abroad 행의 면적 칸은 원본에 점선만 있다. 0으로 채우지 않고 빈 값(—)으로 두었다.
- Population 열은 제목이 '1910'이지만 Philippine Islands는 1903년, Panama Canal Zone은 1912년 값이며 Guam·Samoa는 추정치다(원본 각주·행 이름). CSV의 population_note 열에 남겼다.
- OCR 점수(score)는 정답 확률이 아니다: 가장 큰 오류(17,635,426)의 점수가 0.996이었고, 맞게 읽은 6,449의 점수는 0.808이었다.
- 열 경계(세로 괘선)와 행 군집 허용오차(11px), 제목/각주 영역 경계(y=100/455px)는 이 표 이미지에 맞춘 규칙이다. 다른 표에는 그대로 적용되지 않을 수 있다.
