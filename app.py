import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from st_aggrid import AgGrid, GridOptionsBuilder
 

st.set_page_config(
    page_title = '관극 정산',
    layout = 'wide'
)

alt.themes.enable('dark')


전체 = pd.read_excel('RawData.xlsx')


#폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'


#-------------------------------------------------------------------------------------------------------------------------------------------------------------


#필터

year_list = sorted(전체['날짜'].dt.year.unique(), reverse=True)
display_list = ['전체'] + [f"{year}년 관극 정산" for year in year_list]

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    selected_option = st.selectbox('', display_list, index=0)


tbl_review = {
    2018: "https://docs.google.com/document/d/1ngNlvLkkbjqr2dKBcf6GSFSng658vXWsZWSRmY0p_EI/edit?tab=t.0",
    2019: "https://docs.google.com/document/d/1ngNlvLkkbjqr2dKBcf6GSFSng658vXWsZWSRmY0p_EI/edit?tab=t.0#heading=h.cgexjii66zaa",
    2020: "https://docs.google.com/document/d/1IJKj28WSmB43cowlF19ppH67wi0aAqRkUBKykEZYuqo/edit?tab=t.0",
    2021: "https://docs.google.com/document/d/11HkLPS0i1BaHYxd_nrXF-DIXJddcswiEALKDXXaMLyk/edit?tab=t.0",
    2022: "https://docs.google.com/document/d/14ko3jNxFIZktf9Vp6kEA59ANtVfHfnph4zFvlagN6m8/edit?tab=t.0",
    2023: "https://docs.google.com/document/d/1Lp8e0cK20w1skw57uWFSe5gkgxcq_WRyIVKUViLO08c/edit?tab=t.0",
    2024: "https://docs.google.com/document/d/1PlCMWPXI8RFP3FU5QmjoMivZdmflKNWw6sUZKOcfaEE/edit?tab=t.0",
    2025: "https://docs.google.com/document/d/151p6Aup0qZki6hvL_E73D-XKB7jHAqBNW0_aKy_5m9w/edit?tab=t.0"
}

tbl_cast = {
    2018: "https://x.com/playnmusical_Q/status/1499335369652772864",
    2019: "https://x.com/playnmusical_Q/status/1499341967087394817",
    2020: "https://x.com/playnmusical_Q/status/1219481959170068480",
    2021: "https://x.com/playnmusical_Q/status/1352549590537408513",
    2022: "https://x.com/playnmusical_Q/status/1478714300579876866",
    2023: "https://x.com/playnmusical_Q/status/1681233081586552834",
    2024: "https://x.com/playnmusical_Q/status/1742747279763648953",
    2025: "https://x.com/playnmusical_Q/status/1875163614967202161"
}


if selected_option != '전체':
    selected_year = int(selected_option.split('년')[0])
    후기_링크 = tbl_review.get(selected_year, '')
    캐보_링크 = tbl_cast.get(selected_year, '')
else:
    후기_링크 = ''
    캐보_링크 = ''



if selected_option == '전체':
    ''
else:
    with col3:
        col_btn1, col_btn2 = st.columns([1, 3.4])

        with col_btn1:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            st.link_button("캐슷 보드", 캐보_링크)

        with col_btn2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            st.link_button("후기", 후기_링크)



selected_year = year_list[display_list.index(selected_option) - 1]


if selected_option == '전체':
    df_selected_year = 전체
    df_year_before = 전체
else: 
    df_selected_year = 전체[전체['날짜'].dt.year == selected_year]
    df_year_before = 전체[전체['날짜'].dt.year == (selected_year - 1)]


#-------------------------------------------------------------------------------------------------------------------------------------------------------------


#월별
df_selected_year = df_selected_year.copy()
df_selected_year['날짜'] = pd.to_datetime(df_selected_year['날짜'])
df_selected_year['가격'] = df_selected_year['가격'].replace(',', '', regex=True).astype(int)

df_selected_year['가격'] = pd.to_numeric(df_selected_year['가격'], errors='coerce')
df_selected_year['가격'] = df_selected_year['가격'].fillna(0)
df_selected_year['가격'] = df_selected_year['가격'].astype(int)

df_selected_year['년도'] = df_selected_year['날짜'].dt.year
df_selected_year['월'] = df_selected_year['날짜'].dt.month


연도들 = df_selected_year['년도'].unique()
기준표 = pd.DataFrame([
    {'년도': y, '월': m}
    for y in 연도들
    for m in range(1, 13)
])


집계표 = (
    df_selected_year
    .groupby(['년도', '월'])
    .agg(
        횟수=('극', 'count'),
        극=('극', 'nunique'),
        비용=('가격', 'sum')
    )
    .reset_index()
)


월별정산 = pd.merge(기준표, 집계표, on=['년도', '월'], how='left').fillna(0)
월별정산[['횟수', '극', '비용']] = 월별정산[['횟수', '극', '비용']].astype(int)

월별정산 = 월별정산.sort_values(by=['년도', '월']).reset_index(drop=True)



#------------
year_list = df_selected_year['년도'].unique()

year = []
for x in year_list:
    filtered = df_selected_year[df_selected_year['년도'] == x]
    show = filtered['극'].nunique()
    count = filtered.shape[0]
    expense = filtered['가격'].sum()
    year.append({
        '년도': x,
        '극': show,
        '횟수': count,
        '비용': expense
    })


연별정산 = pd.DataFrame(year)



#------------



#그래프
if selected_option == '전체':
    x = np.arange(len(연별정산))
    x_labels = [f'\n{y}년' for y in 연별정산['년도'].values]
    횟수 = 연별정산['횟수'].values
    극 = 연별정산['극'].values
    비용 = 연별정산['비용'].values
else:
    x = np.arange(1, 13)  # 1~12
    x_labels = [f'\n{m}월' for m in x]
    횟수 = 월별정산['횟수'].values
    극 = 월별정산['극'].values
    비용 = 월별정산['비용'].values

fig0, ax1 = plt.subplots(figsize=(13.6, 3.8))
fig0.patch.set_facecolor('none')
ax1.set_facecolor('none')


# '횟수'
bar_width = 0.45 
cmap = LinearSegmentedColormap.from_list("bar_grad", ["#BED3C8", "#68B676", "#00780E"])

for xi, y in zip(x, 횟수):
    grad = np.linspace(0, 1, 100).reshape(-1, 1)
    ax1.imshow(
        grad,
        aspect='auto',
        extent=(xi - bar_width/2, xi + bar_width/2, 0, y),
        origin='lower',
        cmap=cmap,
        zorder=2
    )

# '극'
for xi, y in zip(x, 극):
    ax1.bar(
        xi, y, width=bar_width,
        fill=False,
        edgecolor='#F0F0F0',
        linewidth=1.5,
        linestyle=(0, (5, 5)),
        zorder=3
    )

# '비용'
ax2 = ax1.twinx()
ax2.plot(
    x, 비용,
    color='#C0C0C0',
    linewidth=2.25,
    marker=None,
    zorder=4
)
ax2.set_ylim(bottom=0)

ax1.set_xticks(x)
ax1.set_xticklabels(x_labels, fontsize=12, color='#AFABAB')
ax1.tick_params(axis='x', length=0, labelsize=12, colors='#AFABAB')
ax1.tick_params(axis='y', left=False, labelleft=False)
ax2.tick_params(axis='y', right=False, labelright=False)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_color('#C0C0C0')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)

ax1.grid(False)
ax2.grid(False)


# Legend
legend_elements = [
    Patch(facecolor='#68B676', edgecolor='none', label='횟수'),
    Patch(facecolor='none', edgecolor='#F0F0F0', linewidth=1, linestyle=(0, (5, 5)), label='극'),
    Line2D([0], [0], color='#C0C0C0', lw=2.25, label='비용')
]
leg = ax1.legend(
    handles=legend_elements,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.3),
    ncol=3,
    frameon=False,
    fontsize=10,
    labelcolor='#AFABAB'
)

# Data Labels
for xi, y in zip(x, 횟수):
    if y > 0:
        ax1.text(
            xi, y + max(횟수)*0.07, f"{y}",
            ha='center', va='bottom',
            fontsize=8, color='#E0E0E0', zorder=10
        )
for xi, y in zip(x, 극):
    if y > 0:
        ax1.text(
            xi, 0.5, f"{y}",
            ha='center', va='bottom',
            fontsize=8, color='#404040', zorder=10
        )


plt.subplots_adjust(left=0.12, right=0.93, bottom=0.18, top=0.83)




#-------------------------------------------------------------------------------------------------------------------------------------------------------------


#장르
genre_list = ['연극', '뮤지컬', '기타']

genre = []
for x in genre_list:
    count = df_selected_year['장르'].str.contains(x, na=False).sum()
    genre.append({
        '장르': x,
        '횟수': count
    })

장르 = pd.DataFrame(genre)

장르 = 장르[장르['횟수'] > 0].reset_index(drop=True)


#장르 도넛 차트
fig, ax = plt.subplots(figsize = (5,5))
fig.patch.set_facecolor('none')

ax.pie(
    장르['횟수'],
    autopct='%.0f%%',
    counterclock=False,
    startangle=90,
    radius=1,
    colors=['#228B22','#5DC35D','#A4FBA6'],
    pctdistance=0.82,
    wedgeprops=dict( #각 조각 설정
        width=0.35,
        edgecolor='#3B3838',
        linewidth=2
        ),
    textprops=dict(
        color = '#4D4D4D',
        fontsize = 11,
        fontweight = 'bold'
        )
       )


#총 횟수
올해횟수 = 장르['횟수'].sum()
data = pd.read_excel('/Users/js/Desktop/RawData.xlsx')
작년횟수 = data[data['날짜'].dt.year == (selected_year-1)].shape[0]
차이 = 올해횟수 - 작년횟수

if 차이 > 0:
    차이표시 = f'▲ {차이}'
    색상 = '#FD6666'
elif 차이 < 0:
    차이표시 = f'▼ {abs(차이)}'
    색상 = '#6FB3FF'
else:
    차이표시 = '-'
    색상 = '#BFBFBF'



if selected_option == '전체': 
    ax.text(0,0.0,f'{올해횟수}회', ha='center', va='center', fontsize = 40, fontweight = 'bold', color = 'white')
    ''
else:
    ax.text(0,-0.25,차이표시, ha='center', va='center', fontsize = 18, fontweight = 'bold', color = 색상)
    ax.text(0,0.05,f'{올해횟수}회', ha='center', va='center', fontsize = 40, fontweight = 'bold', color = 'white')


#legend
ax.legend(
    장르['장르'],
    loc = 'lower center',
    bbox_to_anchor = (0.5, -0.15),
    ncol = len(장르),
    frameon = False,
    fontsize = 12,
    labelcolor='#D9D9D9'
)


#-------------------------------------------------------------------------------------------------------------------------------------------------------------


#회전 표
극 = df_selected_year['극'].value_counts().to_frame().reset_index()
극.columns = ['극', '횟수']

nth_list = ['자첫자막', '자둘', '자셋+']

nth = []
for x in nth_list:
    filtered = 극['횟수'].apply(
        lambda x: '자첫자막' if x == 1 else ('자둘' if x == 2 else '자셋+'))
    show = (filtered == x).sum()
    nth.append({
        '회전': x,
        '극': show
    })

회전 = pd.DataFrame(nth)


#장르 도넛 차트
fig2, ax = plt.subplots(figsize = (5,5))
fig2.patch.set_facecolor('none')

ax.pie(
    회전['극'],
    autopct='%.0f%%',
    counterclock=False,
    startangle=90,
    radius=1,
    colors=['#228B22','#5DC35D','#A4FBA6'],
    pctdistance=0.82,
    wedgeprops=dict( #각 조각 설정
        width=0.35,
        edgecolor='#3B3838',
        linewidth=2
        ),
    textprops=dict(
        color = '#4D4D4D',
        fontsize = 11,
        fontweight = 'bold'
        )
       )


#총 횟수
올해극 = 회전['극'].sum()
data = pd.read_excel('/Users/js/Desktop/RawData.xlsx')
작년극 = data[data['날짜'].dt.year == (selected_year-1)]['극'].nunique()
차이 = 올해극 - 작년극

if 차이 > 0:
    차이표시 = f'▲ {차이}'
    색상 = '#FD6666'
elif 차이 < 0:
    차이표시 = f'▼ {abs(차이)}'
    색상 = '#6FB3FF'
else:
    차이표시 = '-'
    색상 = '#BFBFBF'


if selected_option == '전체': 
    ax.text(0,0,f'{올해극}극', ha='center', va='center', fontsize = 40, fontweight = 'bold', color = 'white')
    ''
else:
    ax.text(0,0.05,f'{올해극}극', ha='center', va='center', fontsize = 40, fontweight = 'bold', color = 'white')
    ax.text(0,-0.25,차이표시, ha='center', va='center', fontsize = 18, fontweight = 'bold', color = 색상)


#legend
ax.legend(
    회전['회전'],
    loc = 'lower center',
    bbox_to_anchor = (0.5, -0.15),
    ncol = len(회전),
    frameon = False,
    fontsize = 12,
    labelcolor='#D9D9D9'
)


#-------------------------------------------------------------------------------------------------------------------------------------------------------------


#본사극 분석
창작초 = df_selected_year['시즌'].str.contains('창작 초연|창작 트아').sum()
창작재 = df_selected_year['시즌'].str.contains('창작 재연').sum()
창작삼 = df_selected_year['시즌'].str.contains('창작').sum() - 창작초 - 창작재

라센초 = df_selected_year['시즌'].str.contains('라센 초연|라센 트아').sum()
라센재 = df_selected_year['시즌'].str.contains('라센 재연').sum()
라센삼 = df_selected_year['시즌'].str.contains('라센').sum() - 라센초 - 라센재


season = [
    {'회차': '초연 & 트아', '창작': 창작초, '라센': 라센초},
    {'회차': '재연', '창작': 창작재, '라센': 라센재},
    {'회차': '삼연 이상', '창작': 창작삼, '라센': 라센삼}
]

시즌 = pd.DataFrame(season)





#시즌 그래프
labels = ['창작', '라센']
categories = ['초연 & 트아', '재연', '삼연 이상']

창작 = 시즌['창작'].values
라센 = 시즌['라센'].values

y = np.arange(len(labels))
창작_values = [시즌.loc[시즌['회차'] == cat, '창작'].values[0] for cat in categories]
라센_values = [시즌.loc[시즌['회차'] == cat, '라센'].values[0] for cat in categories]

bar_width = 0.45

fig3, ax3 = plt.subplots(figsize=(5, 5))
fig3.patch.set_facecolor('none')
ax3.set_facecolor('none')

#스택
p1 = ax3.barh(labels, [창작_values[0], 라센_values[0]], bar_width, label='초연 & 트아', color='#228B22', edgecolor='#3B3838', linewidth=2)
p2 = ax3.barh(labels, [창작_values[1], 라센_values[1]], bar_width, left=[창작_values[0], 라센_values[0]], label='재연', color='#5DC35D', edgecolor='#3B3838', linewidth=2)
p3 = ax3.barh(labels, [창작_values[2], 라센_values[2]], bar_width,
             left=[창작_values[0]+창작_values[1], 라센_values[0]+라센_values[1]],
             label='삼연 이상', color='#A4FBA6', edgecolor='#3B3838', linewidth=2)

ax3.set_ylim(-0.5, len(labels)-0.5)
ax3.set_yticklabels([label + '  ' for label in labels], color='#D9D9D9', fontsize=15, fontweight='bold')
ax3.invert_yaxis()


#Legend
ax3.legend(
    ['초연', '재연', '삼연+'],
    loc = 'lower center',
    bbox_to_anchor = (0.5, -0.345),
    ncol = len(categories),
    frameon = False,
    fontsize = 14,
    labelcolor='#D9D9D9'
)


#Data Labels
for p in [p1, p2, p3]:
    custom_labels = [f'{int(val)}' if val != 0 else '' for val in p.datavalues]
    ax3.bar_label(
        p,
        labels=custom_labels,
        label_type='center',
        color='#4D4D4D',
        fontsize=11,
        fontweight='bold'
    )


ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_color('#C0C0C0')
ax3.spines['bottom'].set_visible(False)

ax3.set_xticks([])


#-------------------------------------------------------------------------------------------------------------------------------------------------------------


# 배우
actor_list = df_selected_year['캐슷'].str.strip().str.split(' ').explode().unique()

actor = []
for x in actor_list:
    filtered = df_selected_year['캐슷'].apply(lambda y: (' '+x+' ') in y)
    count = filtered.sum()
    nshow = df_selected_year[filtered]['극'].nunique()
    actor.append({'배우': x, '횟수': count, '필모': nshow})


배우 = pd.DataFrame(actor)

배우['배우'] = pd.Categorical(배우['배우'],
                          categories=actor_list,
                          ordered=True)

배우 = 배우.sort_values(by=['횟수', '필모', '배우'], ascending=[False, False, True])

배우['배우'] = 배우['배우'].apply(lambda x: x.replace('_', ' ') if '_' in x else x)
배우['배우'] = 배우['배우'].apply(lambda x: x.replace('[', ' [') if '[' in x else x)
배우['배우'] = 배우['배우'].apply(lambda x: x.replace('(', ' (') if '(' in x else x)


actor_diff = 배우.shape[0] - df_year_before['캐슷'].str.strip().str.split(' ').explode().nunique()

if actor_diff > 0:
    차이표시 = f'▲ {actor_diff}'
    색상 = '#FD6666'
elif actor_diff < 0:
    차이표시 = f'▼ {abs(actor_diff)}'
    색상 = '#6FB3FF'
else:
    차이표시 = '-'
    색상 = '#BFBFBF'



if selected_option == '전체': 
    배우_text = f'<p style="font-size: 17px;">{"&nbsp;" * 2}{배우.shape[0]}명</p>'
else:
    배우_text = f'<p style="font-size: 17px;">{"&nbsp;" * 2}{배우.shape[0]}명  (<span style="font-size: 12px; color: {색상};">{차이표시}</span>)</p>'



#Column width
actor_grid = GridOptionsBuilder.from_dataframe(배우)

actor_grid.configure_column('배우', width=140, filter=True) 
actor_grid.configure_column('횟수', width=80)
actor_grid.configure_column('필모', width=80)


#-------------------------------------------------------------------------------------------------------------------------------------------------------------


# 극
극 = df_selected_year['극'].value_counts().to_frame().reset_index()
극.columns=['극', '횟수']

show_list = df_selected_year['극'].unique()
극['극'] = pd.Categorical(극['극'],
                        categories=show_list,
                        ordered=True
)

극 = 극.sort_values(by=['횟수','극'], ascending=[False, True])


연극_극수 = df_selected_year[df_selected_year['장르'].str.contains('연극')]['극'].nunique()
뮤지컬_극수 = df_selected_year[df_selected_year['장르'].str.contains('뮤지컬')]['극'].nunique()
기타_극수 = df_selected_year[df_selected_year['장르'].str.contains('기타')]['극'].nunique()

기타_항목 = f', 기타 {기타_극수}극' if 기타_극수 > 0 else ''
극_text = f'<p style="font-size: 17px;">{"&nbsp;" * 2}: 연극 {연극_극수}극, 뮤지컬 {뮤지컬_극수}극{기타_항목}</p>'



show_grid = GridOptionsBuilder.from_dataframe(극)

show_grid.configure_column('극', width=220, filter=True) 
show_grid.configure_column('횟수', width=80)


#-------------------------------------------------------------------------------------------------------------------------------------------------------------

if selected_option == '전체': 
    텍스트 = '연별 정산'
else:
    텍스트 = '월별 정산'


# Layout
left, right = st.columns([3, 2])

with left:

    with st.container():
        st.markdown(f'<p style="font-size: 20px;">《 {텍스트} 》</p>', unsafe_allow_html=True)
        st.pyplot(fig0)
        st.markdown(f'<div style="height:10px"></div>', unsafe_allow_html=True)

    left_col1, left_col2, left_col3 = st.columns(3)
    with left_col1:
        st.markdown(f'<p style="font-size: 20px;">《 장르 비율 》</p>', unsafe_allow_html=True)
        st.pyplot(fig)
    with left_col2:
        st.markdown(f'<p style="font-size: 20px;">《 회전률 》</p>', unsafe_allow_html=True)
        st.pyplot(fig2)
    with left_col3:
        st.markdown(f'<p style="font-size: 20px;">《 시즌 》</span></p>', unsafe_allow_html=True)
        st.pyplot(fig3)
    

fixed_height = 555

with right:
    right_up1, right_up2 = st.columns(2)
    with right_up1:
        st.markdown(f'<p style="font-size: 20px;">《 배우 》</p>', unsafe_allow_html=True)
        st.markdown(배우_text, unsafe_allow_html=True)
        AgGrid(
            배우,
            height=fixed_height,
            fit_columns_on_grid_load=True,
            theme='streamlit',
            gridOptions=actor_grid.build()
        )
    with right_up2:
        st.markdown(f'<p style="font-size: 20px;">《 작품 》</p>', unsafe_allow_html=True)
        st.markdown(극_text, unsafe_allow_html=True)
        AgGrid(
            극,
            height=fixed_height,
            fit_columns_on_grid_load=True,
            theme='streamlit',
            gridOptions=show_grid.build()
        )

