import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import requests

# DATAFRAME DE BASE #
data = pd.read_csv('data_global.csv')
data.dropna(subset='genres', inplace=True)
# api key
my_api_key = '1e58c94b59a1d6b57eff4872239e559f'

st.set_page_config(layout='wide')

# Affichage du logo au-dessus de la sidebar
logo_path = "logo3.JPG"
st.sidebar.image(
                logo_path,
                use_container_width=True
                 )


def get_poster_url(movie_id, api_key):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": api_key}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        poster_path = response.json().get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
        return None
    else:
        st.write(movie_id)

# --- INSTANCIATION DU MENU VIA ST.SIDEBAR ---#
with st.sidebar:
    selection = (
            option_menu
                (
                        menu_title='',
                        options=[
                            'ETUDES ET STATISTIQUES',
                            'RECOMMANDATIONS DE FILMS ANNEE 2024',
                            'RECOMMANDATIONS DE FILMS DEPUIS 1960',
                            'ACTEUR / ACTRICE',
                            'REALISATEUR',
                            'DRAME',
                            'COMEDIE',
                            "ACTION / AVENTURE",
                            'THRILLER / CRIME',
                            'RECHERCHE PAR FILM'
                        ]
                    )
                )


# ---------------- HOMEPAGE ----------------#
# sera affiché sur la homepage, une recommandation
# ==> les trois films les mieux notés, etc...
if selection == 'ETUDES ET STATISTIQUES':
    st.title("Périmètre d'études")
    st.subheader('', divider='orange')
    st.write('')
    st.image('Pop age creuse.jpg', use_container_width=True)
    st.divider()
    st.write('')
    st.image('Nb hab csp.JPG', use_container_width=True)
    st.divider()
    st.write('')
    st.image('films tranches age sup 35 ans.JPG', use_container_width=True)
    st.divider()
    st.write('')
    st.image('Nb films par genre.PNG', use_container_width=False)


# ----------- RECOMMANDATIONS DE FILMS --------------------------#
# 12 meilleurs films, genres sélectionnées par nous memes, de 2023, ayant au moins 100K votes

elif selection == 'RECOMMANDATIONS DE FILMS ANNEE 2024':
    st.title("Les 12 films plébiscités en 2024")
    st.write('')
    # on fait une copie du dataframe
    data2 = data
    data2 = data2[data2['genres'].str.contains('Drama|Comedy|Thriller|Action|Crime|Adventure')]
    data2.drop_duplicates(subset='tconst', inplace=True)

    accueil = data2[data2['startYear'] == 2024][['tconst', 'title', 'startYear', 'numVotes', 'averageRating']]
    acceuil = accueil[accueil['numVotes'] > 100000]

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.image(get_poster_url(acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[0][0], my_api_key)
                 , caption=acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[0][1])
    with col2:
        st.image(get_poster_url(acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[1][0], my_api_key)
                 , caption=acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[1][1])
    with col3:
        st.image(get_poster_url(acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[2][0], my_api_key)
                 , caption=acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[2][1])
    with col4:
        st.image(get_poster_url(acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[3][0], my_api_key)
                 , caption=acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[3][1])
    with col5:
        st.image(get_poster_url(acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[4][0], my_api_key)
                 , caption=acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[4][1])
    with col6:
        st.image(get_poster_url(acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[5][0], my_api_key)
                 , caption=acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[5][1])
    col7, col8, col9, col10, col11, col12 = st.columns(6)
    with col7:
        st.image(get_poster_url(acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[6][0], my_api_key)
                , caption=acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[6][1])
    with col8:
        st.image(get_poster_url(acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[7][0], my_api_key)
                , caption=acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[7][1])
    with col9:
        st.image(get_poster_url(acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[8][0], my_api_key)
                , caption=acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[8][1])
    with col10:
        st.image(get_poster_url(acceuil.sort_values(by = 'averageRating', ascending=False).head(13)[['tconst', 'title']].values[9][0], my_api_key)
                , caption=acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[9][1])
    with col11:
        st.image(get_poster_url(acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[10][0], my_api_key)
                , caption=acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[10][1])
    with col12:
        st.image(get_poster_url(acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[11][0], my_api_key)
                , caption=acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[11][1])

elif selection == 'RECOMMANDATIONS DE FILMS DEPUIS 1960':
    st.title("Les 12 films plébiscités depuis 1960")
    st.write('')

    data2 = data
    data2 = data2[data2['genres'].str.contains('Drama|Comedy|Thriller|Action|Crime|Adventure')]

    data2.drop_duplicates(subset='tconst', inplace=True)

    accueil = data2[data2['startYear'] >= 1960][['tconst', 'title', 'startYear', 'numVotes', 'averageRating']]
    acceuil = accueil[accueil['numVotes'] > 100000]

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.image(get_poster_url(acceuil.sort_values(by = 'averageRating', ascending=False).head(13)[['tconst', 'title']].values[0][0], my_api_key)
                 , caption=acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[0][1])
    with col2:
        st.image(get_poster_url(acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[1][0], my_api_key)
                 , caption=acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[1][1])
    with col3:
        st.image(get_poster_url(acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[2][0], my_api_key)
                 , caption=acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[2][1])
    with col4:
        st.image(get_poster_url(acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[3][0], my_api_key)
                 , caption=acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[3][1])
    with col5:
        st.image(get_poster_url(acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[4][0], my_api_key)
                 , caption=acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[4][1])
    with col6:
        st.image(get_poster_url(acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[5][0], my_api_key)
                 , caption=acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[5][1])
    col7, col8, col9, col10, col11, col12 = st.columns(6)
    with col7:
        st.image(get_poster_url(acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[6][0], my_api_key)
                , caption=acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[6][1])
    with col8:
        st.image(get_poster_url(acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[7][0], my_api_key)
                , caption=acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[7][1])
    with col9:
        st.image(get_poster_url(acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[8][0], my_api_key)
                , caption=acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[8][1])
    with col10:
        st.image(get_poster_url(acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[9][0], my_api_key)
                , caption=acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[9][1])
    with col11:
        st.image(get_poster_url(acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[10][0], my_api_key)
                , caption=acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[10][1])
    with col12:
        st.image(get_poster_url(acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[11][0], my_api_key)
                , caption=acceuil.sort_values(by='averageRating', ascending=False).head(13)[['tconst', 'title']].values[11][1])


# ----- SELECTION D'UN ACTEUR/ ACTRICE EN PARTICULIER -----------#
# ---------------- CHOIX DE L'ACTEUR PAR USER -------------------#
# --------- totalité de la BDD pour choix de user ---------------#
elif selection == "ACTEUR / ACTRICE":  # insertion par user
    st.title("Sélectionnez un acteur ou une actrice pour afficher tous leurs films")

    st.subheader('', divider='orange')

    # on instancie le dataframe actor servant de base choix user
    df_actor = data.loc[(data['category'].isin(['actor', 'actress']))].sort_values(by='numVotes', ascending=False)

    # df_actor.rename(columns={'averageRating': 'Note moyenen'}, inplace=True)

    df_actor.rename(columns={'title': 'Titre',
                             'averageRating': 'Note moyenne',
                             'numVotes': 'Votes',
                             'startYear': 'Année',
                             'genres': 'Genre(s)',
                             'runtimeMinutes': 'Durée(mn)',
                             },
                    inplace=True
                    )
    # liste pour ordre d'apparition des colonnes du dataframe
    list_columns_df_actor = ['Titre',
                             'Note moyenne',
                             'Votes',
                             'Année',
                             'Genre(s)',
                             'Durée(mn)']

    # créer une liste acteur (unique) pour selectbox
    liste_actor = set(df_actor['primaryName'].str.split(',').explode().to_list())

    for num, ele in enumerate(liste_actor):
        if ele == 'Bruce Willis':
            actor = num

    # création de la selectbox qui recherche le choix de user dans liste_actor
    choix_acteur = st.selectbox("", liste_actor, index=actor)

    st.write('')

    # affichage du dataframe filtré cf. selectbox
    acteur_selectionne = df_actor.loc[df_actor['primaryName'] == choix_acteur]
    st.dataframe(
        acteur_selectionne,
        width=700,
        height=500,
        hide_index=True,
        use_container_width=True,
        column_order=list_columns_df_actor,
        )

# --------- SELECTION D'UN REALISATEUR EN PARTICULIER -------------#
# ---------------- CHOIX DU REALISATEUR PAR USER ------------------#
# ----------- totalité de la BDD pour choix de user ---------------#
elif selection == 'REALISATEUR':
    st.title('Choisissez un réalisateur')

    st.subheader('', divider='orange')

    # on instancie le dataframe réalisateur servant de base choix user
    df_real = data[['primaryName',
                    'title',
                    'startYear',
                    'numVotes',
                    'averageRating',
                    'genres',
                    'tconst']][data['category'] == 'director'].sort_values(by='averageRating', ascending=False)

    df_real.rename(columns={'title': 'Titre',
                             'averageRating': 'Note moyenne',
                             'numVotes': 'Votes',
                             'startYear': 'Année',
                             'genres': 'Genre(s)',
                             'runtimeMinutes': 'Durée(mn)',
                            }, inplace=True
                   )
    
    # liste pour ordre d'apparition des colonnes du dataframe
    list_columns_df_real = ['Titre',
                            'Note moyenne',
                            'Votes',
                            'Année',
                            'Genre(s)',
                            'Durée(mn)']

    # créer une liste realisateurs (unique) pour selectbox
    liste_real = set(df_real['primaryName'].to_list())

    for num, ele in enumerate(liste_real):
        if ele == 'Quentin Tarantino':
            real = num

    # création de la selectbox qui recherche le choix de user dans liste_real
    choix_real = st.selectbox('', liste_real, index=real)

    st.write('')

    # affichage du dataframe filtré cf. selectbox
    real_selectionne = df_real.loc[df_real['primaryName'] == choix_real]
    st.dataframe(
        real_selectionne,
        width=400,
        height=500,
        hide_index=True,
        use_container_width=True,
        column_order=list_columns_df_real
        )

# ------------------ SELECTION FILM DRAME -------------------#
elif selection == 'DRAME':
    st.title('Choisissez un film du genre Drame')

    st.subheader('', divider='orange')

    # ici on affiche le dataframe
    # et la selectbox pour que le user puisse choisir son film
    # on instancie le dataframe genre drama servant de base choix user
    df_drama = data[['title',
                     'genres',
                     'startYear',
                     'runtimeMinutes',
                     'averageRating',
                     'numVotes',
                     'tconst']][data['genres'].str.contains('Drama')].drop_duplicates().sort_values(by='numVotes', ascending=False)

    df_drama.rename(columns={'title': 'Titre',
                             'averageRating': 'Note moyenne',
                             'numVotes': 'Votes',
                             'startYear': 'Année',
                             'genres': 'Genre(s)',
                             'runtimeMinutes': 'Durée(mn)',
                             },
                    inplace=True
                    )
    
    # liste pour ordre d'apparition des colonnes du dataframe
    list_columns_drama = ['Titre',
                          'Note moyenne',
                          'Votes',
                          'Année',
                          'Genre(s)',
                          'Durée(mn)']

    # créer une liste de films genre drama (unique) pour selectbox
    list_films_drama = set(data['title'][data['genres'].str.contains('Drama')])

    for num, ele in enumerate(list_films_drama):
        if ele == 'Pulp Fiction':
            drame = num

    # création de la selectbox qui recherche le choix de user dans liste_drama
    choix_drama = st.selectbox('', list_films_drama, index=drame)

    st.write('')

    # affichage du dataframe filtré cf. selectbox
    film_selectionne_drama = df_drama.loc[df_drama['Titre'] == choix_drama]
    st.dataframe(
        film_selectionne_drama,
        width=800,
        height=100,
        hide_index=True,
        use_container_width=True,
        column_order=list_columns_drama
        )

    col1, col2 = st.columns(2, gap='medium')

    # ici, requête à l'API TMDb pour afficher les posters et synopsis du film en fonction du choix user
    # si choix user film drama, on utilise l'API TMDB pour récupérer l'affiche
    with col1:
        if choix_drama:
            # on déclare dans le if la clé API

            api_key = my_api_key
            # l'url de l'API pour récupérer le poster
            # c'est le endpoint auquel on rajoute search/movie pour accéder
            # renvoie un id
            # possible de faire l endpoint sans search ==> a tester au besoin
            url = f"https://api.themoviedb.org/3/search/movie"
            # params = liste des paramètres nécessaires pour requêter l'API

            params = {
                "query": choix_drama,  # choix_action_aventure = le titre sélectionné dans la selectbox par user
                "api_key": api_key,  # ma clé API
                "language": "fr-FR"}  # pour afficher (si possible) des résultats en français

            # ici on envoie la requête à l'API avec requests.get ==> url + params
            response = requests.get(url, params=params)

            # on vérifie que la requête à bien fonctionné ==> 200 = ok
            if response.status_code == 200:
                # si requête réussie ==> on récupère les résultats de la recherche ==> film correspondant au choix user
                data_tmdb = response.json().get('results', [])
            else:
                st.write("Erreur lors de la requête à TMDb. Vérifiez votre connexion ou clé API.")  # message si problème de connexion ou de clé API

            if data_tmdb:  # data tmdb continet toutes les informations nécessaires/ voulues selon notre besoin
                film_data = data_tmdb[0]  # on prend le premier résultat de la liste
                film_id = film_data['id']  # on récupère l'id du film choisi par user
                synopsis = film_data.get('overview')  # overview fait partie des champs retournés dans la réponse JSON
                st.subheader('Synopsis', divider='orange')
                st.write(synopsis or 'Aucun synopsis disponible')
            else:
                st.write("Aucune information trouvée pour ce film.")  # message si pas d'info trouvée pour le film"

    with col2:
        if data_tmdb:
            poster_path = film_data.get('poster_path')
        else:
            st.write("Pas d'affiche disponible.")

        if poster_path:
            url_affiche = f"https://image.tmdb.org/t/p/w500{poster_path}"  # instanciation de l'url complète de l'affiche ==> endpoint + varialbe poster_path"
            st.image(url_affiche, caption=choix_drama, use_container_width=True)  # on affiche le poster
        else:
            st.write("Aucune affiche trouvée pour ce film.")  # message si pas d'affiche trouvée"

    st.divider()

    headers = {
            "accept": "application/json",
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJmNzliMjE5OTY2YjFiYTczNDliMTFiNjQxNWQ2ZGFjZiIsIm5iZiI6MTczNDU5NjIxNi45NTM5OTk4LCJzdWIiOiI2NzYzZDY3ODU4MWEzYzA1MDdhYjBjODIiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.ep8YcNVjt4GmmtNlO6wYBoBJxfTNwVjs5Ug0B0PuMKI"
        }

    url_video = f"https://api.themoviedb.org/3/movie/{film_id}/videos"
    response_video = requests.get(url_video, params={'api_key': my_api_key, 'language': 'fr-FR'})
    if response_video.status_code == 200:
        video_data = response_video.json().get('results', [])
        trailers = [
            ele for ele in video_data if ele['type'] == 'Trailer' and ele['site'] == 'YouTube'
            ]
        if trailers:
            trailer_key = trailers[0]['key']
            youtube_url = f"https://www.youtube.com/watch?v={trailer_key}"
            st.subheader('Bande-annonce:')
            st.video(youtube_url)
        else:
            st.write('Aucune bande-annonce disponible pour ce film')


# ------------------ SELECTION FILM COMEDIE -------------------#
elif selection == 'COMEDIE':
    st.title('Choisissez une comédie')

    st.subheader('', divider='orange')

    # st.image('photo drame.JPG', width=500)

    # ici on affiche le dataframe et la selectbox
    # pour que le user puisse choisir son film
    # on instancie le dataframe genre drama servant de base choix user
    df_comedie = df_comedie = data[['title',
                                    'startYear',
                                    'runtimeMinutes',
                                    'averageRating',
                                    'numVotes',
                                    'tconst']][data['genres'].str.contains('Comedy')].drop_duplicates().sort_values(by= 'numVotes', ascending= False)

    df_comedie.rename(columns={'title': 'Titre',
                               'averageRating': 'Note moyenne',
                               'numVotes': 'Votes',
                               'startYear': 'Année',
                               'genres': 'Genre(s)',
                               'runtimeMinutes': 'Durée(mn)',
                             }, inplace=True
                      )
    
    # liste pour ordre d'apparition des colonnes du dataframe
    list_columns_comedie = ['Titre',
                            'Note moyenne',
                            'Votes',
                            'Année',
                            'Genre(s)',
                            'Durée(mn)']

    # créer une liste de films genre comedie (unique) pour selectbox
    list_films_comedie = set(data['title'][data['genres'].str.contains('Comedy')])

    for num, ele in enumerate(list_films_comedie):
        if ele == 'Mon voisin le tueur':
            comedie = num

    # création de la selectbox qui recherche le choix de user dans liste_drama
    choix_comedie = st.selectbox('', list_films_comedie, index=comedie)

    st.write('')

    # affichage du dataframe filtré cf. selectbox
    film_selectionne_comedie = df_comedie.loc[df_comedie['Titre'] == choix_comedie]
    st.dataframe(
        film_selectionne_comedie,
        width=800,
        height=100,
        hide_index=True,
        use_container_width=True,
        column_order=list_columns_comedie,
            )

    col1, col2 = st.columns(2, gap='medium')

    # # ici, requête à l'API TMDb pour afficher les posters
    # et synopsis du film en fonction du choix user
    # # si choix user film drama
    # on utilise l'API TMDB pour récupérer l'affiche
    with col1:
        if choix_comedie:
            # on déclare dans le if la clé API

            api_key = my_api_key
            # l'url de l'API pour récupérer le poster
            # c'est le endpoint auquel on rajoute search/movie pour accéder
            # renvoie un id
            # possible de faire l endpoint sans search ==> a tester au besoin
            url = f"https://api.themoviedb.org/3/search/movie"

            # params = liste des paramètres nécessaires pour requêter l'API
            params = {
                "query": choix_comedie,  # choix_action_aventure = le titre sélectionné dans la selctbox par user
                "api_key": api_key,  # ma clé API
                "language": "fr-FR"}  # pour afficher (si possible) des résultats en français

            # ici on envoie la requête à l'API avec requests.get ==> url + params
            response = requests.get(url, params=params)

            # on vérifie que la requête à bien fonctionné ==> 200 = ok
            if response.status_code == 200:
                # si requête réussie ==> on récupère les résultats de la recherche ==> film correspondant au choix user
                data_tmdb = response.json().get('results', [])
            else:  # message si problème de connexion ou de clé API
                st.write("Erreur lors de la requête à TMDb. Vérifiez votre connexion ou clé API.")

            if data_tmdb:  # data tmdb continet toutes les informations nécessaires/ voulues selon notre besoin
                film_data = data_tmdb[0]  # on prend le premier résultat de la liste
                film_id = film_data['id']  # on récupère l'id du film choisi par user
                synopsis = film_data.get('overview')  # overview fait partie des champs retournés dans la réponse JSON
                st.subheader('Synopsis', divider='orange')
                st.write(synopsis or 'Aucun synopsis disponible')
            else:  # message si pas d'info trouvée pour le film"
                st.write("Aucune information trouvée pour ce film.")

    with col2:
        if data_tmdb:
            poster_path = film_data.get('poster_path')
        else:
            st.write("Pas d'affiche disponible.")

        if poster_path:
            url_affiche = f"https://image.tmdb.org/t/p/w500{poster_path}"  # instanciation de l'url complète de l'affiche ==> endpoint + varialbe poster_path"
            st.image(url_affiche, caption=choix_comedie, use_container_width=True)  # on affiche le poster
        else:
            st.write("Aucune affiche trouvée pour ce film.")  # message si pas d'affiche trouvée"

    st.divider()

    headers = {
            "accept": "application/json",
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJmNzliMjE5OTY2YjFiYTczNDliMTFiNjQxNWQ2ZGFjZiIsIm5iZiI6MTczNDU5NjIxNi45NTM5OTk4LCJzdWIiOiI2NzYzZDY3ODU4MWEzYzA1MDdhYjBjODIiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.ep8YcNVjt4GmmtNlO6wYBoBJxfTNwVjs5Ug0B0PuMKI"
        }

    url_video = f"https://api.themoviedb.org/3/movie/{film_id}/videos"
    response_video = requests.get(url_video, params={'api_key': my_api_key, 'language': 'fr-FR'})
    if response_video.status_code == 200:
        video_data = response_video.json().get('results', [])
        trailers = [
            ele for ele in video_data if ele['type'] == 'Trailer' and ele['site'] == 'YouTube'
            ]
        if trailers:
            trailer_key = trailers[0]['key']
            youtube_url = f"https://www.youtube.com/watch?v={trailer_key}"
            st.subheader('Bande-annonce:')
            st.video(youtube_url)
        else:
            st.write('Aucune bande-annonce disponible pour ce film')


# ------------------ SELECTION FILM ACTION/ AVENTURE -------------------#
elif selection == "ACTION / AVENTURE":
    st.title("Choisissez un film d'Action / Aventure")

    st.subheader('', divider='orange')

    # on instancie le dataframe genre action servant de base choix user
    df_action_adventure = data[['title',
                                'startYear',
                                'runtimeMinutes',
                                'averageRating',
                                'numVotes',
                                'tconst']][data['genres'].str.contains('Action','Adventure')].drop_duplicates().sort_values(by='numVotes', ascending=False)

    df_action_adventure.rename(columns={'title': 'Titre',
                                        'averageRating': 'Note moyenne',
                                        'numVotes': 'Votes',
                                        'startYear': 'Année',
                                        'genres': 'Genre(s)',
                                        'runtimeMinutes': 'Durée(mn)',
                                        }, inplace=True
                               )
    # liste pour ordre d'apparition des colonnes du dataframe
    list_columns_action_aventure = ['Titre',
                                    'Note moyenne',
                                    'Votes',
                                    'Année',
                                    'Genre(s)',
                                    'Durée(mn)']

    # créer une liste de films genre action (unique) pour selectbox
    list_films_action_adventure = set(data['title'][data['genres'].str.contains('Action', 'Adventure')])

    for num, ele in enumerate(list_films_action_adventure):
        if ele == 'Le Cinquième Élément':
            actadv = num

    # qui recherche le choix de user dans list_action_adventure
    choix_action_aventure = st.selectbox("", list_films_action_adventure, index=actadv)

    st.write('')

    # affichage du dataframe filtré cf. selectbox
    film_selectionne_action_aventure = df_action_adventure.loc[df_action_adventure['Titre'] == choix_action_aventure]
    st.dataframe(
        film_selectionne_action_aventure,
        width=400,
        height=100,
        hide_index=True,
        use_container_width=True,
        column_order=list_columns_action_aventure
        )

    col1, col2 = st.columns(2, gap='medium')

    with col1:
        if choix_action_aventure:
            # on déclare dans le if la clé API
            api_key = my_api_key
            # l'url de l'API pour récupérer le poster
            # c'est le endpoint auquel on rajoute search/movie pour accéder
            # renvoie un id
            # possible de faire l endpoint sans search ==> a tester au besoin
            url = f"https://api.themoviedb.org/3/search/movie"
            # params = liste des paramètres nécessaires pour requêter l'API
            params = {
                "query": choix_action_aventure,  # choix_action_aventure = le titre sélectionné dans la selctbox par user
                "api_key": api_key,  # ma clé API
                "language": "fr-FR"}  # pour afficher (si possible) des résultats en français

            # ici on envoie la requête à l'API avec requests.get ==> url + params
            response = requests.get(url, params=params)

            # on vérifie que la requête à bien fonctionné ==> 200 = ok
            if response.status_code == 200:
                # si requête réussie ==> on récupère les résultats de la recherche ==> film correspondant au choix user   
                data_tmdb = response.json().get('results', [])
            else:
                st.write("Erreur lors de la requête à TMDb. Vérifiez votre connexion ou clé API.")  # message si problème de connexion ou de clé API

            if data_tmdb:  # data tmdb continet toutes les informations nécessaires/ voulues selon notre besoin
                film_data = data_tmdb[0]  # on prend le premier résultat de la liste
                film_id = film_data['id']  # on récupère l'id du film choisi par user
                synopsis = film_data.get('overview')  # overview fait partie des champs retournés dans la réponse JSON
                st.subheader('Synopsis', divider='orange')
                st.write(synopsis or 'Aucun synopsis disponible')
            else:
                film_id = None  # on declare = None pour ne pas avoir d'erreur
                st.write("Aucune information trouvée pour ce film.")  # message si pas d'info trouvée pour le film"

    with col2:
        poster_path = None  # on declare = None pour ne pas avoir d'erreur
        if data_tmdb:
            poster_path = film_data.get('poster_path')
        else:
            st.write("Pas d'affiche disponible.")

        if poster_path:
            url_affiche = f"https://image.tmdb.org/t/p/w500{poster_path}"  # instanciation de l'url complète de l'affiche ==> endpoint + varialbe poster_path"
            st.image(url_affiche, caption=choix_action_aventure, use_container_width=True)  # on affiche le poster
        else:
            st.write("Aucune affiche trouvée pour ce film.")  # message si pas d'affiche trouvée"

    st.divider()

    headers = {
            "accept": "application/json",
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJmNzliMjE5OTY2YjFiYTczNDliMTFiNjQxNWQ2ZGFjZiIsIm5iZiI6MTczNDU5NjIxNi45NTM5OTk4LCJzdWIiOiI2NzYzZDY3ODU4MWEzYzA1MDdhYjBjODIiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.ep8YcNVjt4GmmtNlO6wYBoBJxfTNwVjs5Ug0B0PuMKI"
        }

    url_video = f"https://api.themoviedb.org/3/movie/{film_id}/videos"
    response_video = requests.get(url_video, params={'api_key': my_api_key, 'language': 'fr-FR'})
    if response_video.status_code == 200:
        video_data = response_video.json().get('results', [])
        trailers = [
            ele for ele in video_data if ele['type'] == 'Trailer' and ele['site'] == 'YouTube'
            ]
        if trailers:
            trailer_key = trailers[0]['key']
            youtube_url = f"https://www.youtube.com/watch?v={trailer_key}"
            st.subheader('Bande-annonce:')
            st.video(youtube_url)
        else:
            st.write('Aucune bande-annonce disponible pour ce film')


# ------------------ SELECTION FILM THRILLER / CRIME -------------------#
elif selection == 'THRILLER / CRIME':
    st.title('Choisissez un Thriller / Crime')
    st.subheader('', divider='orange')

    # ici on affiche le dataframe et la selectbox pour que le user puisse choisir son film
    # on instancie le dataframe genre thriller / crime servant de base choix user
    df_thriller_crime = data[['title',
                              'startYear',
                              'runtimeMinutes',
                              'averageRating',
                              'numVotes',
                              'tconst']][data['genres'].str.contains('Thriller','Crime')].drop_duplicates().sort_values(by='numVotes', ascending=False)

    df_thriller_crime.rename(columns={'title': 'Titre',
                                      'averageRating': 'Note moyenne',
                                      'numVotes': 'Votes',
                                      'startYear': 'Année',
                                      'genres': 'Genre(s)',
                                      'runtimeMinutes': 'Durée(mn)',
                                      }, inplace=True
                             )
    # liste pour ordre d'apparition des colonnes du dataframe
    list_columns_thriller_crime = ['Titre',
                                   'Note moyenne',
                                   'Votes',
                                   'Année',
                                   'Genre(s)',
                                   'Durée(mn)']

    # créer une liste de films genre comedie (unique) pour selectbox
    list_films_thriller_crime = set(data['title'][data['genres'].str.contains('Thriller', 'Crime')])

    for num, ele in enumerate(list_films_thriller_crime):
        if ele == '58 Minutes pour vivre':
            thricri = num

    # création de la selectbox qui recherche le choix de user dans liste_drama
    choix_thriller_crime = st.selectbox('', list_films_thriller_crime, index=thricri)

    st.write('')

    # affichage du dataframe filtré cf. selectbox
    film_selectionne_thriller_crime = df_thriller_crime.loc[df_thriller_crime['Titre'] == choix_thriller_crime]
    st.dataframe(
        film_selectionne_thriller_crime,
        width=800,
        height=100,
        hide_index=True,
        use_container_width=True,
        column_order=list_columns_thriller_crime
        )

    col1, col2 = st.columns(2, gap='medium')

    # # ici, requête à l'API TMDb pour afficher les posters et synopsis du film en fonction du choix user
    # # si choix user film drama, on utilise l'API TMDB pour récupérer l'affiche
    with col1:
        if choix_thriller_crime:
            # on déclare dans le if la clé API
            api_key = my_api_key
            # l'url de l'API pour récupérer le poster
            # c'est le endpoint auquel on rajoute search/movie pour accéder
            # renvoie un id
            # possible de faire l endpoint sans search ==> a tester au besoin
            url = f"https://api.themoviedb.org/3/search/movie"
            # params = liste des paramètres nécessaires pour requêter l'API
            params = {
                "query": choix_thriller_crime,  # choix_action_aventure = le titre sélectionné dans la selctbox par user
                "api_key": api_key,  # ma clé API
                "language": "fr-FR"}  # pour afficher (si possible) des résultats en français

            # ici on envoie la requête à l'API avec requests.get ==> url + params
            response = requests.get(url, params=params)

            # on vérifie que la requête à bien fonctionné ==> 200 = ok
            if response.status_code == 200:
                # si requête réussie ==> on récupère les résultats de la recherche ==> film correspondant au choix user   
                data_tmdb = response.json().get('results', [])
            else:
                st.write("Erreur lors de la requête à TMDb. Vérifiez votre connexion ou clé API.")  # message si problème de connexion ou de clé API

            if data_tmdb:  # data tmdb continet toutes les informations nécessaires/ voulues selon notre besoin
                film_data = data_tmdb[0]  # on prend le premier résultat de la liste
                film_id = film_data['id']  # on récupère l'id du film choisi par user
                synopsis = film_data.get('overview')  # overview fait partie des champs retournés dans la réponse JSON
                st.subheader('Synopsis', divider='orange')
                st.write(synopsis or 'Aucun synopsis disponible')
            else:
                st.write("Aucune information trouvée pour ce film.")  # message si pas d'info trouvée pour le film"

    with col2:
        if data_tmdb:
            poster_path = film_data.get('poster_path')
        else:
            st.write("Pas d'affiche disponible.")

        if poster_path:
            url_affiche = f"https://image.tmdb.org/t/p/w500{poster_path}"  # instanciation de l'url complète de l'affiche ==> endpoint + varialbe poster_path"
            st.image(url_affiche, caption=choix_thriller_crime, use_container_width=True)  # on affiche le poster
        else:
            st.write("Aucune affiche trouvée pour ce film.")  # message si pas d'affiche trouvée"

    st.divider()

    headers = {
            "accept": "application/json",
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJmNzliMjE5OTY2YjFiYTczNDliMTFiNjQxNWQ2ZGFjZiIsIm5iZiI6MTczNDU5NjIxNi45NTM5OTk4LCJzdWIiOiI2NzYzZDY3ODU4MWEzYzA1MDdhYjBjODIiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.ep8YcNVjt4GmmtNlO6wYBoBJxfTNwVjs5Ug0B0PuMKI"
        }

    url_video = f"https://api.themoviedb.org/3/movie/{film_id}/videos"
    response_video = requests.get(url_video, params={'api_key': my_api_key, 'language': 'fr-FR'})
    if response_video.status_code == 200:
        video_data = response_video.json().get('results', [])
        trailers = [
            ele for ele in video_data if ele['type'] == 'Trailer' and ele['site'] == 'YouTube'
            ]
        if trailers:
            trailer_key = trailers[0]['key']
            youtube_url = f"https://www.youtube.com/watch?v={trailer_key}"
            st.subheader('Bande-annonce:')
            st.video(youtube_url)
        else:
            st.write('Aucune bande-annonce disponible pour ce film')


# ----------------- TOUS LES FILMS ----------------#
elif selection == 'RECHERCHE PAR FILM':
    st.title("Tous les films")

    st.subheader('', divider='orange')

    df_all = data[['title',
                   'genres',
                   'startYear',
                   'runtimeMinutes',
                   'averageRating',
                   'numVotes',
                   'tconst']].drop_duplicates().sort_values(by='numVotes', ascending=False)

    df_all.rename(columns={'title': 'Titre',
                           'averageRating': 'Note moyenne',
                           'numVotes': 'Votes',
                           'startYear': 'Année',
                           'genres': 'Genre(s)',
                           'runtimeMinutes': 'Durée(mn)',
                           }, inplace=True
                  )
    
    # liste pour ordre d'apparition des colonnes du dataframe
    list_columns_df_all = ['Titre',
                           'Note moyenne',
                           'Votes',
                           'Année',
                           'Genre(s)',
                           'Durée(mn)']

    list_films_all = set(data['title'])

    for num, ele in enumerate(list_films_all):
        if ele == 'Sixième Sens':
            film3 = num

    choix_films_all = st.selectbox('Choisissez', list_films_all, label_visibility='hidden', index=film3)

    film_selectionne_all = df_all.loc[df_all['Titre'] == choix_films_all]
    st.dataframe(
        film_selectionne_all,
        width=800,
        height=100,
        hide_index=True,
        use_container_width=True,
        column_order=list_columns_df_all
        )

    col1, col2 = st.columns(2, gap='medium')

    # # ici, requête à l'API TMDb pour afficher les posters et synopsis du film en fonction du choix user
    # # si choix user film drama, on utilise l'API TMDB pour récupérer l'affiche
    with col1:
        if choix_films_all:
            # on déclare dans le if la clé API
            api_key = my_api_key
            # l'url de l'API pour récupérer le poster
            # c'est le endpoint auquel on rajoute search/movie pour accéder
            # renvoie un id
            # possible de faire l endpoint sans search ==> a tester au besoin
            url = f"https://api.themoviedb.org/3/search/movie"
            # params = liste des paramètres nécessaires pour requêter l'API
            params = {
                "query": choix_films_all,  # choix_action_aventure = le titre sélectionné dans la selctbox par user
                "api_key": api_key,  # ma clé API
                "language": "fr-FR"}  # pour afficher (si possible) des résultats en français

            # ici on envoie la requête à l'API avec requests.get ==> url + params
            response = requests.get(url, params=params)

        # on vérifie que la requête à bien fonctionné ==> 200 = ok
        if response.status_code == 200:
            # si requête réussie ==> on récupère les résultats de la recherche ==> film correspondant au choix user   
            data_tmdb = response.json().get('results', [])
        else:
            st.write("Erreur lors de la requête à TMDb. Vérifiez votre connexion ou clé API.")  # message si problème de connexion ou de clé API

        if data_tmdb:  # data tmdb continet toutes les informations nécessaires/ voulues selon notre besoin
            film_data = data_tmdb[0]  # on prend le premier résultat de la liste
            film_id = film_data['id']  # on récupère l'id du film choisi par user
            synopsis = film_data.get('overview')  # overview fait partie des champs retournés dans la réponse JSON
            st.subheader('Synopsis', divider='orange')
            st.write(synopsis or 'Aucun synopsis disponible')
        else:
            st.write("Aucune information trouvée pour ce film.")  # message si pas d'info trouvée pour le film"

    with col2:
        if data_tmdb:
            poster_path = film_data.get('poster_path')
        else:
            st.write("Pas d'affiche disponible.")

        if poster_path:
            url_affiche = f"https://image.tmdb.org/t/p/w500{poster_path}"  # instanciation de l'url complète de l'affiche ==> endpoint + varialbe poster_path"
            st.image(url_affiche, caption=choix_films_all, use_container_width=True)  # on affiche le poster
        else:
            st.write("Aucune affiche trouvée pour ce film.")  # message si pas d'affiche trouvée"

    st.divider()

    url_video = f"https://api.themoviedb.org/3/movie/{film_id}/videos"
    response_video = requests.get(url_video, params={'api_key': my_api_key, 'language': 'fr-FR'})
    if response_video.status_code == 200:
        video_data = response_video.json().get('results', [])
        trailers = [
            ele for ele in video_data if ele['type'] == 'Trailer' and ele['site'] == 'YouTube'
            ]
        if trailers:
            trailer_key = trailers[0]['key']
            youtube_url = f"https://www.youtube.com/watch?v={trailer_key}"
            st.subheader('Bande-annonce:')
            st.video(youtube_url)
        else:
            st.write('Aucune bande-annonce disponible pour ce film')
