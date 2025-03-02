import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import requests

# DATAFRAME DE BASE #
data = pd.read_csv('data_global.csv')

# api key
my_api_key = '1e58c94b59a1d6b57eff4872239e559f'

st.set_page_config(layout='wide')


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


def get_video_url(movie_id, api_key):
    url_video = f"https://api.themoviedb.org/3/movie/{movie_id}/videos"
    response_video = requests.get(url_video, params={'api_key': my_api_key, 'language': 'fr-FR'})

    if response_video.status_code == 200:
        video_data = response_video.json().get('results', [])

        trailers = [
            ele for ele in video_data
            if ele['type'] == 'Trailer' and ele['site'] == 'YouTube'
            ]

        if trailers:
            trailer_key = trailers[0]['key']
            youtube_url = f"https://www.youtube.com/watch?v={trailer_key}"
            return (youtube_url)


# Affichage du logo au-dessus de la sidebar
logo_path = "logo3.JPG"
st.sidebar.image(
                logo_path,
                use_container_width=True
                 )

# --- INSTANCIATION DU MENU VIA ST.SIDEBAR ---#
with st.sidebar:
    selection = (
            option_menu(
                        menu_title='',
                        options=[
                            'ACCUEIL',
                            'RECOMMANDATION DE FILMS',
                            "RECHERCHE PAR ACTEUR",
                            'RECHERCHE DE FILM',
                        ]
                    )
                )


# ---------------- HOMEPAGE ----------------#
# sera affiché sur la homepage, une recommandation ==> les trois films les mieux notés, etc...
if selection == 'ACCUEIL':
    st.title("Bienvenu(e) sur le site des Cinéphiles")
    st.subheader('A ne pas rater !', divider='orange')
    st.write('')
    # on fait une copie du dataframe
    data2 = data
    data2.drop_duplicates(subset='tconst', inplace=True)
    accueil = data2[data2['startYear'] == 2024][['tconst', 'title', 'startYear', 'numVotes', 'averageRating']]
    acceuil = accueil[accueil['numVotes'] > 100000]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(get_poster_url(acceuil.sort_values(by='averageRating', ascending=False).head(3)[['tconst', 'title']].values[0][0], my_api_key),
                 caption=acceuil.sort_values(by='averageRating', ascending=False).head(3)[['tconst', 'title']].values[0][1])
        col4, col5, col6 = st.columns(3)
        with col4:
            pass
        with col5:
            trailer_1 = st.button('Trailer ')  # attention j'ai laissé des blancs différents pour différencier les boutton si ils sont tous trailer ça fait une erreur
        with col6:
            pass
    with col2:
        st.image(get_poster_url(acceuil.sort_values(by='averageRating', ascending=False).head(3)[['tconst', 'title']].values[1][0], my_api_key),
                 caption=acceuil.sort_values(by='averageRating', ascending=False).head(3)[['tconst', 'title']].values[1][1])
        col4, col5, col6 = st.columns(3)
        with col4:
            pass
        with col5:
            trailer_2 = st.button('Trailer')  # attention j'ai laissé des blancs différents pour différencier les boutton si ils sont tous trailer ça fait une erreur
        with col6:
            pass
    with col3:
        st.image(get_poster_url(acceuil.sort_values(by='averageRating', ascending=False).head(3)[['tconst', 'title']].values[2][0], my_api_key),
                 caption=acceuil.sort_values(by='averageRating', ascending=False).head(3)[['tconst', 'title']].values[2][1])
        col4, col5, col6 = st.columns(3)
        with col4:
            pass
        with col5:
            trailer_3 = st.button('Trailer  ')  # attention j'ai laissé des blancs différents pour différencier les boutton si ils sont tous trailer ça fait une erreur
        with col6:
            pass
    st.subheader('', divider='orange')
    for val, ele in enumerate([trailer_1, trailer_2, trailer_3]):
        if ele:
            film_id = acceuil.sort_values(by='averageRating', ascending=False).head(3)[['tconst', 'title']].values[val][0]
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


# --------- SELECTION D'UN FILM EN PARTICULIER ------------------#
# ---------------- MACHINE LEARNING -----------------------------#
elif selection == 'RECOMMANDATION DE FILMS':
    st.title('Recherche par film')

    st.subheader('', divider='orange')

    # DATAFRAME MACHINE LEARNING
    df_ml = pd.read_csv('df_ml_1')

    list_df_ml_movie = set(df_ml['title'])

    list_columns_df_ml = ['title',
                          'averageRating',
                          'runtimeMinutes']

    for num, ele in enumerate(list_df_ml_movie):
        if ele == 'Le Cinquième Élément':
            film = num

    choix_film_df_ml = st.selectbox('', list_df_ml_movie, index=film)

    movie_selectionne = df_ml.loc[df_ml['title'] == choix_film_df_ml]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(df_ml.drop(columns=['tconst','title']))

    # on applique notre jeu de donnée au modèle pour l'entrainer
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=4, metric='euclidean')
    nn.fit(X_scaled)

    # Recherche des films

    _index = df_ml[df_ml['title'] == choix_film_df_ml].index[0]
    distances, indices = nn.kneighbors(X_scaled[_index].reshape(1, -1))

    list_ml = []
    list_ml_titre = []

    for distance, index in zip(distances[0][1:], indices[0][1:]):  # On exclut ce film en commençant à 1
        list_ml.append(df_ml['tconst'].iloc[index])
        list_ml_titre.append(df_ml['title'].iloc[index])

    st.write('')
    st.write('##### **Nous vous recommandons la sélection suivante:**')
    st.write('')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(get_poster_url(list_ml[0], my_api_key), caption=list_ml_titre[0])
        col4, col5, col6 = st.columns(3)
        with col4:
            pass
        with col5:
            trailer_1 = st.button('Trailer ')  # attention j'ai laissé des blancs différents pour différencier les boutton si ils sont tous trailer ça fait une erreur
        with col6:
            pass
    with col2:
        st.image(get_poster_url(list_ml[1], my_api_key), caption=list_ml_titre[1])
        col4, col5, col6 = st.columns(3)
        with col4:
            pass
        with col5:
            trailer_2 = st.button('Trailer')  # attention j'ai laissé des blancs différents pour différencier les boutton si ils sont tous trailer ça fait une erreur
        with col6:
            pass
    with col3:
        st.image(get_poster_url(list_ml[2], my_api_key), caption=list_ml_titre[2])
        col4, col5, col6 = st.columns(3)
        with col4:
            pass
        with col5:
            trailer_3 = st.button('Trailer  ')  # attention j'ai laissé des blancs différents pour différencier les boutton si ils sont tous trailer ça fait une erreur
        with col6:
            pass

    for val, ele in enumerate([trailer_1, trailer_2, trailer_3]):
        if ele:
            film_id = list_ml[val]
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


# ----- SELECTION D'UN ACTEUR/ ACTRICE EN PARTICULIER -----------#
# ---------------- CHOIX DE L'ACTEUR PAR USER -------------------#
elif selection == "RECHERCHE PAR ACTEUR":  # insertion par user
    st.title("Recherchez votre acteur")
    st.subheader('', divider='orange')

    # on instancie le dataframe actor servant de base choix user
    df_actor = data.loc[(data['category'].isin(['actor', 'actress']))].sort_values(by='numVotes', ascending=False)

    # liste pour ordre d'apparition des colonnes du dataframe
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
                column_order=list_columns_df_actor)

# ----------------- TOUS LES FILMS ----------------#
elif selection == 'RECHERCHE DE FILM':
    st.title("Recherchez votre film")
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

    list_columns_df_all = ['Titre',
                           'Note moyenne',
                           'Votes',
                           'Genre(s)',
                           'Durée(mn)']

    list_films_all = set(data['title'])

    for num, ele in enumerate(list_films_all):
        if ele == 'Piège de cristal':
            film_2 = num

    choix_films_all = st.selectbox('Choisissez', list_films_all, label_visibility='hidden', index=film_2)

    st.write('')

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

        if data_tmdb:  # data tmdb contient toutes les informations nécessaires/ voulues selon notre besoin
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
