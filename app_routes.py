from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_file
import pandas as pd
import numpy as np
import os
import json
import tempfile
import uuid
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import sqlite3

# Importer nos modules personnalisés
from modules.db_connection import DatabaseManager
from modules.data_processor_module import DataProcessor
from modules.clustering_module import ClusteringProcessor
from modules.visualization_module import create_visualization, generate_report
from modules.history_manager_module import AnalysisHistory, PDFAnalysisHistory
from modules.transformations_persistence import TransformationManager

import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration de l'application
app = Flask(__name__)
app.secret_key = "milan_app_secret_key_2025"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limite 16 Mo
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'txt', 'xlsx', 'pdf', 'wav', 'mp3'}

# S'assurer que le dossier d'upload existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialisation des gestionnaires
db_manager = DatabaseManager('init_database/fidelity_db')
data_processor = DataProcessor()
transformation_manager = TransformationManager('uploads')
history_manager = AnalysisHistory('analysis_history')
pdf_history_manager = PDFAnalysisHistory('analysis_history/pdf')

# Routes principales
@app.route('/')
def index():
    """Page d'accueil de l'application"""
    return render_template('index.html')

# Modifie la route data_processing pour gérer correctement les filtres
@app.route('/data_processing', methods=['GET', 'POST'])
def data_processing():
    """
    Page de traitement de données CSV ou accès à la base de données
    Gère le chargement et le filtrage des transactions depuis SQLite
    """
    # Initialisation des variables de connexion et d'état
    db_connected = False
    db_info = ""
    db_error = "Base de données non initialisée."
    transactions_count = 0
    
    # Chemin de la base de données
    db_path = 'modules/fidelity_db.sqlite'
    
    # Variables pour stocker les données des listes déroulantes
    points_vente = []
    enseignes = []
    categories_produits = []
    villes = []
    produits = []
    moyens_paiement = []
    
    def decompose_transaction_with_articles(transaction_data):
        """
        Décompose un enregistrement de transaction avec ses articles.
        
        Args:
            transaction_data (dict): Dictionnaire contenant les détails de la transaction et ses articles
        
        Returns:
            list: Liste de dictionnaires, un pour chaque article de la transaction
        """
        # Extraire les détails de base de la transaction
        base_details = {
            'id': transaction_data['id'],
            'date_transaction': transaction_data['date_transaction'],
            'montant_total': transaction_data['montant_total'],
            'numero_facture': transaction_data['numero_facture'],
            'magasin': transaction_data['magasin'],
            'enseigne': transaction_data['enseigne'],
            'moyen_paiement': transaction_data['moyen_paiement'],
            'canal_vente': transaction_data['canal_vente'],
            'points_gagnes': transaction_data['points_gagnes']
        }
        
        # Si pas d'articles, retourner un seul enregistrement avec des détails d'article vides
        if not transaction_data.get('articles', []):
            base_details.update({
                'nom_article': None,
                'quantite': None,
                'prix_unitaire': None,
                'remise_pourcentage': None,
                'montant_ligne': None,
                'categorie': None
            })
            return [base_details]
        
        # Créer un enregistrement pour chaque article de la transaction
        decomposed_records = []
        for article in transaction_data['articles']:
            record = base_details.copy()
            record.update({
                'nom_article': article.get('nom_article'),
                'quantite': article.get('quantite'),
                'prix_unitaire': article.get('prix_unitaire'),
                'remise_pourcentage': article.get('remise_pourcentage'),
                'montant_ligne': article.get('montant_ligne'),
                'categorie': article.get('categorie')
            })
            decomposed_records.append(record)
        
        return decomposed_records

    # Bloc de connexion et récupération des données
    try:
        # Afficher le chemin complet de la base de données
        abs_db_path = os.path.abspath(db_path)
        
        # Vérifier si le fichier existe
        if os.path.exists(db_path):
            # Connexion à la base de données
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Tester la connexion en récupérant la liste des tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            table_names = [table[0] for table in tables]
            table_count = len(table_names)
            
            # Vérifier si les tables nécessaires existent
            has_points_vente = 'points_vente' in table_names
            has_transactions = 'transactions' in table_names
            has_produits = 'produits' in table_names
            has_categories = 'categories_produits' in table_names
            
            # Récupérer des informations sur les transactions
            if has_transactions:
                try:
                    cursor.execute("SELECT COUNT(*) as count FROM transactions")
                    transactions_count = cursor.fetchone()[0]
                except Exception as e:
                    logger.error(f"Erreur lors du comptage des transactions: {e}")
                    transactions_count = 0
            
            db_connected = True
            db_info = f"Base de données connectée: {abs_db_path}. {table_count} tables trouvées. {transactions_count} transactions enregistrées."
            
            # Récupération des données pour les listes déroulantes
            if has_points_vente:
                try:
                    # Points de vente
                    cursor.execute("""
                        SELECT magasin_id, nom, ville, code_postal, type, email as enseigne
                        FROM points_vente
                        WHERE statut != 'fermé'
                        ORDER BY nom
                    """)
                    points_vente = [dict(row) for row in cursor.fetchall()]
                    
                    # Enseignes
                    cursor.execute("""
                        SELECT DISTINCT email as enseigne 
                        FROM points_vente 
                        WHERE email IS NOT NULL 
                        ORDER BY email
                    """)
                    enseignes = [row[0] for row in cursor.fetchall() if row[0]]
                    
                    # Villes
                    cursor.execute("""
                        SELECT DISTINCT ville 
                        FROM points_vente 
                        WHERE ville IS NOT NULL 
                        ORDER BY ville
                    """)
                    villes = [row[0] for row in cursor.fetchall() if row[0]]
                except Exception as e:
                    logger.error(f"Erreur lors de la récupération des points de vente: {e}")
            
            # Catégories de produits
            if has_categories:
                try:
                    cursor.execute("""
                        SELECT categorie_id, nom, description, categorie_parent_id
                        FROM categories_produits
                        ORDER BY nom
                    """)
                    categories_produits = [dict(row) for row in cursor.fetchall()]
                except Exception as e:
                    logger.error(f"Erreur lors de la récupération des catégories: {e}")
            
            # Produits
            if has_produits:
                try:
                    cursor.execute("""
                        SELECT produit_id, reference, nom, categorie_id, marque, prix_standard
                        FROM produits
                        WHERE statut = 'actif'
                        ORDER BY nom
                        LIMIT 100
                    """)
                    produits = [dict(row) for row in cursor.fetchall()]
                except Exception as e:
                    logger.error(f"Erreur lors de la récupération des produits: {e}")
            
            # Moyens de paiement
            if has_transactions:
                try:
                    cursor.execute("""
                        SELECT DISTINCT type_paiement
                        FROM transactions
                        WHERE type_paiement IS NOT NULL
                        ORDER BY type_paiement
                    """)
                    moyens_paiement = [row[0] for row in cursor.fetchall() if row[0]]
                except Exception as e:
                    logger.error(f"Erreur lors de la récupération des moyens de paiement: {e}")
            
            conn.close()
        else:
            logger.warning(f"Base de données non trouvée: {abs_db_path}")
            db_error = f"Base de données non trouvée: {abs_db_path}"
    
    except Exception as e:
        logger.error(f"Erreur de connexion à la base de données: {e}")
        db_error = f"Erreur de connexion: {str(e)}"
    
    # Traitement des requêtes POST
    if request.method == 'POST':
        # Vérifier le type de source de données
        data_source = request.form.get('data_source', '')
        
        # Traitement du fichier CSV
        if 'file' in request.files and data_source != 'sqlite':
            file = request.files['file']
            if file.filename == '':
                flash('Aucun fichier sélectionné', 'warning')
                return redirect(request.url)
            
            if file and allowed_file(file.filename):
                # TODO: Implémenter le traitement du fichier CSV
                # Code existant de traitement CSV
                pass
        
        # Traitement des données de la base de données SQLite
        elif data_source == 'sqlite' or (not 'file' in request.files and db_connected):
            try:
                # Récupérer les filtres du formulaire
                filters = {
                    'date_debut': request.form.get('date_debut', ''),
                    'date_fin': request.form.get('date_fin', ''),
                    'magasin_id': request.form.get('magasin_id', ''),
                    'enseigne': request.form.get('enseigne', ''),
                    'ville': request.form.get('ville', ''),
                    'categorie_id': request.form.get('categorie_id', ''),
                    'produit_id': request.form.get('produit_id', ''),
                    'moyen_paiement': request.form.get('moyen_paiement', ''),
                    'montant_min': request.form.get('montant_min', ''),
                    'montant_max': request.form.get('montant_max', ''),
                    'include_items': request.form.get('include_items') == 'true'
                }
                
                logger.info(f"Filtres appliqués: {filters}")
                
                # Gestion des dates par défaut
                if not filters['date_debut'] or not filters['date_fin']:
                    today = datetime.now()
                    filters['date_fin'] = today.strftime('%Y-%m-%d') if not filters['date_fin'] else filters['date_fin']
                    filters['date_debut'] = (today - timedelta(days=90)).strftime('%Y-%m-%d') if not filters['date_debut'] else filters['date_debut']
                
                # Connexion à la base de données
                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Construction de la requête de base
                query = """
                SELECT t.transaction_id as id, t.date_transaction, t.montant_total, 
                       t.numero_facture, pv.nom as magasin, pv.email as enseigne, 
                       t.type_paiement as moyen_paiement, t.canal_vente, t.points_gagnes
                FROM transactions t
                LEFT JOIN points_vente pv ON t.magasin_id = pv.magasin_id
                """
                
                # Ajout des jointures conditionnelles
                if filters['categorie_id'] or filters['produit_id']:
                    query += """
                    JOIN details_transactions dt ON t.transaction_id = dt.transaction_id
                    JOIN produits p ON dt.produit_id = p.produit_id
                    """
                
                # Initialisation des conditions WHERE
                query += " WHERE 1=1"
                params = []
                
                # Ajout des conditions de filtrage
                conditions_filtres = [
                    ('date_transaction', '>=', filters['date_debut']),
                    ('date_transaction', '<=', filters['date_fin']),
                    ('t.magasin_id', '=', filters['magasin_id']),
                    ('pv.email', '=', filters['enseigne']),
                    ('pv.ville', '=', filters['ville']),
                    ('p.categorie_id', '=', filters['categorie_id']),
                    ('dt.produit_id', '=', filters['produit_id']),
                    ('t.type_paiement', '=', filters['moyen_paiement'])
                ]
                
                for colonne, operateur, valeur in conditions_filtres:
                    if valeur:
                        query += f" AND {colonne} {operateur} ?"
                        params.append(valeur)
                
                # Filtres de montant
                if filters['montant_min']:
                    query += " AND t.montant_total >= ?"
                    params.append(float(filters['montant_min']))
                
                if filters['montant_max']:
                    query += " AND t.montant_total <= ?"
                    params.append(float(filters['montant_max']))
                
                # Gestion des doublons
                if filters['categorie_id'] or filters['produit_id']:
                    query += " GROUP BY t.transaction_id"
                
                # Tri et limitation
                query += " ORDER BY t.date_transaction DESC LIMIT 5000"
                
                logger.info(f"Requête SQL: {query}")
                logger.info(f"Paramètres: {params}")
                
                # Exécution de la requête
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                if not rows:
                    flash('Aucune donnée trouvée avec les filtres spécifiés', 'warning')
                    conn.close()
                    return redirect(request.url)
                
                # Conversion en DataFrame
                df = pd.DataFrame([dict(row) for row in rows])
                
                # Traitement des articles si demandé
                if filters['include_items']:
                    try:
                        # Récupération des articles
                        transaction_ids = df['id'].tolist()
                        transaction_ids_str = ','.join(['?' for _ in transaction_ids])
                        
                        items_query = f"""
                        SELECT dt.transaction_id, p.nom as nom_article, dt.quantite, dt.prix_unitaire, 
                               dt.remise_pourcentage, dt.montant_ligne, cp.nom as categorie
                        FROM details_transactions dt
                        JOIN produits p ON dt.produit_id = p.produit_id
                        LEFT JOIN categories_produits cp ON p.categorie_id = cp.categorie_id
                        WHERE dt.transaction_id IN ({transaction_ids_str})
                        """
                        
                        cursor.execute(items_query, transaction_ids)
                        items = [dict(row) for row in cursor.fetchall()]
                        
                        # Groupement des articles par transaction
                        items_by_transaction = {}
                        for item in items:
                            transaction_id = item['transaction_id']
                            if transaction_id not in items_by_transaction:
                                items_by_transaction[transaction_id] = []
                            items_by_transaction[transaction_id].append(item)
                        
                        # Décomposition des transactions
                        decomposed_transactions = []
                        for _, transaction in df.iterrows():
                            transaction_dict = transaction.to_dict()
                            transaction_dict['articles'] = items_by_transaction.get(transaction['id'], [])
                            decomposed_transaction = decompose_transaction_with_articles(transaction_dict)
                            decomposed_transactions.extend(decomposed_transaction)
                        
                        # Mise à jour du DataFrame
                        # Mise à jour du DataFrame
                        df = pd.DataFrame(decomposed_transactions)
                        
                        logger.info(f"Décomposition des transactions : {len(df)} lignes")
                        
                    except Exception as e:
                        logger.error(f"Erreur lors de la décomposition des transactions : {e}")
                
                # Fermeture de la connexion
                conn.close()
                
                # Conversion des types de données
                if 'date_transaction' in df.columns:
                    df['date_transaction'] = pd.to_datetime(df['date_transaction'])
                
                if 'montant_total' in df.columns:
                    df['montant_total'] = pd.to_numeric(df['montant_total'])
                
                # Génération d'un identifiant unique pour le fichier
                file_id = f"db_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Sauvegarde du DataFrame
                transformation_manager.save_original_dataframe(file_id, df)
                
                # Mise à jour de la session
                session['file_id'] = file_id
                session['filename'] = f"transactions_{datetime.now().strftime('%Y%m%d')}"
                
                # Redirection vers la page d'aperçu
                return redirect(url_for('data_preview'))
                
            except Exception as e:
                flash(f'Erreur lors du chargement des données: {str(e)}', 'danger')
                logger.error(f"Erreur lors du chargement des données SQL: {e}")
                return redirect(request.url)
    
    # Méthode GET : préparation du formulaire
    # Dates par défaut
    default_date_fin = datetime.now().strftime('%Y-%m-%d')
    default_date_debut = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    # Récupération des analyses récentes
    recent_analyses = []
    try:
        recent_analyses = history_manager.get_recent_analyses(5)
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des analyses récentes: {e}")
    
    # Rendu du template
    return render_template('data_processing.html', 
                          magasins=points_vente, 
                          enseignes=enseignes,
                          villes=villes,
                          categories_produits=categories_produits,
                          produits=produits,
                          moyens_paiement=moyens_paiement,
                          recent_analyses=recent_analyses,
                          db_connected=db_connected,
                          db_info=db_info,
                          db_error=db_error,
                          tickets_count=transactions_count,
                          default_date_debut=default_date_debut,
                          default_date_fin=default_date_fin)

@app.route('/data_preview')
def data_preview():
    """Page d'aperçu des données"""
    # Vérifier que le fichier est bien chargé
    file_id = session.get('file_id')
    filename = session.get('filename')
    
    if not file_id or not filename:
        flash('Aucune donnée chargée. Veuillez d\'abord charger un fichier CSV ou accéder à la base de données.', 'warning')
        return redirect(url_for('data_processing'))
    
    # Récupérer le DataFrame actuel
    df = transformation_manager.get_current_dataframe(file_id)
    
    if df is None:
        flash('Erreur lors de la récupération des données. Veuillez recharger le fichier.', 'danger')
        return redirect(url_for('data_processing'))
    
    # Préparer les informations du DataFrame pour l'affichage
    df_info = {
        'shape': df.shape,
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_values': df.isna().sum().sum(),
        'has_numeric': len(df.select_dtypes(include=['number']).columns) > 0,
        'numeric_count': len(df.select_dtypes(include=['number']).columns),
        'columns': df.columns.tolist()
    }
    
    # Créer un aperçu HTML des données (limité à 100 lignes)
    preview_data = df.head(100).to_html(classes='table table-striped table-hover', index=False)
    
    return render_template('data_preview.html',
                           df_info=df_info,
                           preview_data=preview_data,
                           filename=filename,
                           columns=df.columns.tolist())

# Dans app_routes.py

@app.route('/api/dataset_preview')
def api_dataset_preview():
    """API pour récupérer les données d'aperçu du dataset"""
    # Vérifier si un fichier est chargé
    file_id = session.get('file_id')
    if not file_id:
        return jsonify({"error": "Aucune donnée chargée"})
    
    # Récupérer le DataFrame
    df = transformation_manager.get_current_dataframe(file_id)
    if df is None:
        return jsonify({"error": "Erreur lors de la récupération des données"})
    
    # Limiter le nombre d'éléments à récupérer
    limit = int(request.args.get('limit', 10))
    
    # Vérifier si le DataFrame contient une colonne d'articles
    has_articles = 'articles' in df.columns
    
    try:
        if has_articles:
            # Extraire les tickets avec leurs articles (limité)
            tickets_data = []
            for _, row in df.head(limit).iterrows():
                ticket = {
                    'id': int(row['id']) if 'id' in row else None,
                    'date_transaction': row['date_transaction'].isoformat() if pd.notna(row.get('date_transaction')) else None,
                    'montant_total': float(row['montant_total']) if pd.notna(row.get('montant_total')) else 0,
                    'magasin': row.get('magasin', 'Inconnu'),
                    'enseigne': row.get('enseigne', None),
                    'moyen_paiement': row.get('moyen_paiement', None),
                    'articles': row.get('articles', [])
                }
                tickets_data.append(ticket)
            
            return jsonify({
                "success": True,
                "tickets": tickets_data,
                "has_articles": True
            })
        else:
            # Pour les DataFrames sans articles
            data = df.head(limit).to_dict('records')
            return jsonify({
                "success": True,
                "data": data,
                "has_articles": False
            })
    
    except Exception as e:
        app.logger.error(f"Erreur lors de la récupération de l'aperçu: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route('/data_transform', methods=['GET', 'POST'])
def data_transform():
    """Page de transformation des données"""
    # Vérifier que le fichier est bien chargé
    file_id = session.get('file_id')
    filename = session.get('filename')
    
    if not file_id or not filename:
        flash('Aucune donnée chargée. Veuillez d\'abord charger un fichier CSV ou accéder à la base de données.', 'warning')
        return redirect(url_for('data_processing'))
    
    # Récupérer le DataFrame original
    df_original = transformation_manager.load_original_dataframe(file_id)
    
    if df_original is None:
        flash('Erreur lors de la récupération des données originales. Veuillez recharger le fichier.', 'danger')
        return redirect(url_for('data_processing'))

    # Pour l'analyse IA
    analysis_result = None
    analysis_pending = False
    
    if request.method == 'POST':
        # Récupérer les transformations demandées
        transformations = request.form.getlist('transformations[]')
        logger.info(f"Transformations demandées: {transformations}")
        
        # Vérifier si c'est une analyse IA
        is_ai_analysis = request.form.get('is_ai_analysis') == 'true'
        
        if is_ai_analysis:
            user_context = request.form.get('user_context', '')
            
            try:
                # Récupérer le DataFrame actuel après toutes les transformations
                current_df = transformation_manager.get_current_dataframe(file_id)
                
                # Importer le transformateur de données qui contient l'accès à l'IA
                from modules.data_transformer_module import DataTransformer
                
                # Créer le transformateur
                data_transformer = DataTransformer()
                
                # Générer l'analyse
                analysis_result = data_transformer.generate_dataset_analysis(current_df, user_context)
                
                # Sauvegarder l'analyse dans l'historique
                if request.form.get('save_history') == 'true':
                    history_manager.add_analysis(
                        dataset_name=filename,
                        dataset_description=f"Analyse IA - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        analysis_text=analysis_result,
                        metadata={
                            "user_context": user_context,
                            "dimensions": current_df.shape,
                            "columns": current_df.columns.tolist(),
                            "transformations": transformation_manager.get_transformations(file_id).get('history', [])
                        }
                    )
                    flash('Analyse IA sauvegardée dans l\'historique', 'success')
                
            except Exception as e:
                logger.error(f"Erreur lors de l'analyse IA: {str(e)}")
                logger.error(traceback.format_exc())
                flash(f'Erreur lors de l\'analyse IA: {str(e)}', 'danger')
                analysis_pending = True
        else:
            # Construire le dictionnaire de transformations
            transform_dict = {}
            
            # Traitement des valeurs manquantes
            if 'missing_values' in transformations:
                strategy = request.form.get('missing_strategy', 'auto')
                transform_dict['missing_values'] = {
                    'strategy': strategy,
                    'threshold': float(request.form.get('missing_threshold', 0.5)),
                    'constant': request.form.get('missing_constant', '')
                }
            
            # Standardisation
            if 'standardization' in transformations:
                columns = request.form.getlist('columns_to_standardize[]')
                transform_dict['standardization'] = {
                    'columns': columns,
                    'method': request.form.get('standardization_method', 'zscore')
                }
            
            # Encodage
            if 'encoding' in transformations:
                columns = request.form.getlist('columns_to_encode[]')
                transform_dict['encoding'] = {
                    'columns': columns,
                    'method': request.form.get('encoding_method', 'one_hot'),
                    'drop_original': request.form.get('drop_original', 'true') == 'true'
                }
            
            # Valeurs aberrantes
            if 'outliers' in transformations:
                columns = request.form.getlist('columns_for_outliers[]')
                transform_dict['outliers'] = {
                    'columns': columns,
                    'method': request.form.get('outlier_detection_method', 'iqr'),
                    'treatment': request.form.get('outlier_treatment', 'tag')
                }
            
            # Ingénierie de caractéristiques
            if 'feature_engineering' in transformations:
                columns = request.form.getlist('interaction_columns[]') if 'interaction_columns[]' in request.form else None
                operations = request.form.getlist('interaction_operations[]') if 'interaction_operations[]' in request.form else None
                transform_dict['feature_engineering'] = {
                    'type_fe': request.form.get('feature_engineering_type', 'interaction'),
                    'columns': columns,
                    'operations': operations
                }
            
            # Suppression de colonnes
            if 'drop_columns' in transformations:
                columns_to_drop = request.form.getlist('columns_to_drop[]')
                transform_dict['drop_columns'] = {
                    'columns_to_drop': columns_to_drop
                }
            
            # Fusion de colonnes
            if 'merge_columns' in transformations:
                columns_to_merge = request.form.getlist('columns_to_merge[]')
                new_column = request.form.get('new_column_name', 'merged_column')
                transform_dict['merge_columns'] = {
                    'columns_to_merge': columns_to_merge,
                    'new_column': new_column,
                    'method': request.form.get('merge_method', 'concat'),
                    'separator': request.form.get('separator', ', '),
                    'drop_original': request.form.get('drop_original_columns', 'false') == 'true'
                }
            
            # Remplacement de valeurs
            if 'replace_values' in transformations:
                column = request.form.get('column_to_replace')
                original_values = request.form.getlist('original_values[]')
                new_values = request.form.getlist('new_values[]')
                
                replacements = {}
                for i in range(len(original_values)):
                    if i < len(new_values) and original_values[i]:
                        replacements[original_values[i]] = new_values[i]
                
                transform_dict['replace_values'] = {
                    'column': column,
                    'replacements': replacements,
                    'replace_all': request.form.get('replace_all_occurrences', 'true') == 'true'
                }
            
            # Appliquer les transformations
            if transform_dict:
                try:
                    # Récupérer le DataFrame actuel
                    current_df = transformation_manager.get_current_dataframe(file_id)
                    
                    # Log avant transformation
                    logger.info(f"DataFrame avant transformation: {current_df.shape}, colonnes: {current_df.columns.tolist()}")
                    
                    # Appliquer les transformations
                    df_transformed, metadata = data_processor.process_dataframe(current_df, transform_dict)
                    
                    # Log après transformation
                    logger.info(f"DataFrame après transformation: {df_transformed.shape}, colonnes: {df_transformed.columns.tolist()}")
                    
                    # Vérifier si la transformation a eu un effet
                    if df_transformed.equals(current_df):
                        logger.warning("ATTENTION: Le DataFrame transformé est identique à l'original. La transformation n'a pas eu d'effet.")
                    
                    # Sauvegarder le DataFrame transformé (copie profonde pour éviter les problèmes de référence)
                    import copy
                    df_to_save = copy.deepcopy(df_transformed)
                    success = transformation_manager.save_transformed_dataframe(file_id, df_to_save)
                    logger.info(f"Sauvegarde du DataFrame transformé: {'succès' if success else 'échec'}")
                    
                    # Vérifier que la sauvegarde a fonctionné
                    verification_df = transformation_manager.load_transformed_dataframe(file_id)
                    if verification_df is None:
                        logger.error("ERREUR CRITIQUE: Le DataFrame transformé n'a pas été correctement sauvegardé!")
                    elif verification_df.equals(df_transformed):
                        logger.info("VÉRIFICATION RÉUSSIE: Le DataFrame transformé a été correctement sauvegardé et rechargé.")
                    else:
                        logger.warning("ATTENTION: Le DataFrame rechargé après sauvegarde est différent de celui transformé!")
                    
                    # Ajouter la transformation à l'historique
                    for transform_type, params in transform_dict.items():
                        transform_success = transformation_manager.add_transformation(file_id, {
                            "type": transform_type,
                            "params": params,
                            "timestamp": datetime.now().isoformat(),
                            "applied_successfully": success
                        })
                        logger.info(f"Ajout de la transformation {transform_type} à l'historique: {transform_success}")
                    
                    flash('Transformations appliquées avec succès !', 'success')
                except Exception as e:
                    logger.error(f"Erreur lors de l'application des transformations: {str(e)}")
                    logger.error(traceback.format_exc())
                    flash(f'Erreur lors de l\'application des transformations: {str(e)}', 'danger')
            
        return redirect(url_for('data_transform'))
    
    # Pour la méthode GET, récupérer le DataFrame courant pour préparer les informations
    current_df = transformation_manager.get_current_dataframe(file_id)
    
    df_info = {
        'shape': current_df.shape,
        'dtypes': current_df.dtypes.astype(str).to_dict(),
        'missing_values': current_df.isna().sum().sum(),
        'has_numeric': len(current_df.select_dtypes(include=['number']).columns) > 0,
        'numeric_count': len(current_df.select_dtypes(include=['number']).columns),
        'columns': current_df.columns.tolist()
    }
    
    # Colonnes numériques et catégorielles pour les formulaires
    numeric_columns = current_df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = current_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Récupérer l'historique des transformations
    transform_history = transformation_manager.get_transformations(file_id)
    
    return render_template('data_transform.html',
                           df_info=df_info,
                           numeric_columns=numeric_columns,
                           categorical_columns=categorical_columns,
                           filename=filename,
                           analysis_result=analysis_result,
                           analysis_pending=analysis_pending,
                           transformations_history=transform_history.get('history', []))

@app.route('/visualizations')
def visualizations():
    """Page de visualisations"""
    # Vérifier que le fichier est bien chargé
    file_id = session.get('file_id')
    filename = session.get('filename')
    
    if not file_id or not filename:
        flash('Aucune donnée chargée. Veuillez d\'abord charger un fichier CSV ou accéder à la base de données.', 'warning')
        return redirect(url_for('data_processing'))
    
    # Récupérer le DataFrame courant
    df = transformation_manager.get_current_dataframe(file_id)
    
    if df is None:
        flash('Erreur lors de la récupération des données. Veuillez recharger le fichier.', 'danger')
        return redirect(url_for('data_processing'))
    
    # Colonnes numériques pour les visualisations
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    return render_template('visualizations.html',
                           columns=df.columns.tolist(),
                           numeric_columns=numeric_columns,
                           filename=filename)

@app.route('/clustering')
def clustering():
    """Page de clustering"""
    # Vérifier que le fichier est bien chargé
    file_id = session.get('file_id')
    filename = session.get('filename')
    
    if not file_id or not filename:
        flash('Aucune donnée chargée. Veuillez d\'abord charger un fichier CSV ou accéder à la base de données.', 'warning')
        return redirect(url_for('data_processing'))
    
    # Récupérer le DataFrame courant
    df = transformation_manager.get_current_dataframe(file_id)
    
    if df is None:
        flash('Erreur lors de la récupération des données. Veuillez recharger le fichier.', 'danger')
        return redirect(url_for('data_processing'))
    
    # Informations sur le DataFrame
    df_info = {
        'shape': df.shape
    }
    
    # Colonnes numériques pour le clustering
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    return render_template('clustering.html',
                           df_info=df_info,
                           numeric_columns=numeric_columns,
                           filename=filename)

@app.route('/run_clustering', methods=['POST'])
def run_clustering():
    """Exécute l'algorithme de clustering sélectionné"""
    # Vérifier que le fichier est bien chargé
    file_id = session.get('file_id')
    filename = session.get('filename')
    
    if not file_id or not filename:
        flash('Aucune donnée chargée. Veuillez d\'abord charger un fichier CSV ou accéder à la base de données.', 'warning')
        return redirect(url_for('data_processing'))
    
    # Récupérer le DataFrame courant
    df = transformation_manager.get_current_dataframe(file_id)
    
    if df is None:
        flash('Erreur lors de la récupération des données. Veuillez recharger le fichier.', 'danger')
        return redirect(url_for('data_processing'))
    
    # Récupérer les paramètres du formulaire
    algorithm = request.form.get('algorithm')
    columns = request.form.getlist('columns[]')
    
    # Paramètres spécifiques à l'algorithme
    params = {}
    
    if algorithm == 'kmeans':
        params['n_clusters'] = int(request.form.get('kmeans-n-clusters', 3))
        params['max_iter'] = int(request.form.get('kmeans-max-iter', 300))
        params['n_init'] = int(request.form.get('kmeans-n-init', 10))
    
    elif algorithm == 'dbscan':
        params['eps'] = float(request.form.get('dbscan-eps', 0.5))
        params['min_samples'] = int(request.form.get('dbscan-min-samples', 5))
    
    elif algorithm == 'hierarchical':
        params['n_clusters'] = int(request.form.get('hierarchical-n-clusters', 3))
        params['affinity'] = request.form.get('hierarchical-affinity', 'euclidean')
        params['linkage'] = request.form.get('hierarchical-linkage', 'ward')
    
    try:
        # Initialiser le processeur de clustering
        clustering_processor = ClusteringProcessor()
        
        # Exécuter le clustering
        clustering_result = clustering_processor.cluster_data(df, algorithm, columns, params)
        
        if not clustering_result["success"]:
            flash(f'Erreur lors du clustering: {clustering_result.get("error", "Erreur inconnue")}', 'danger')
            return redirect(url_for('clustering'))
        
        # Générer un ID unique pour le clustering
        import uuid
        import pickle
        import os
        clustering_id = f"clustering_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
        # Stocker le DataFrame résultant séparément si présent
        if "result_df" in clustering_result:
            transformation_manager.save_transformed_dataframe(file_id, clustering_result["result_df"])
            # Faire une copie de result_df avant de le supprimer
            result_df_copy = clustering_result["result_df"].copy()
            # Supprimer le DataFrame du résultat avant de le sauvegarder
            del clustering_result["result_df"]
        
        # Créer le dossier pour les résultats de clustering s'il n'existe pas
        clustering_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'clustering_results')
        os.makedirs(clustering_folder, exist_ok=True)
        
        # Sauvegarder les résultats du clustering dans un fichier
        clustering_file = os.path.join(clustering_folder, f"{clustering_id}.pkl")
        with open(clustering_file, 'wb') as f:
            pickle.dump(clustering_result, f)
        
        # Stocker uniquement l'ID du clustering dans la session
        session['clustering_id'] = clustering_id
        
        # Générer un résumé textuel du clustering
        cluster_summary = clustering_processor.generate_cluster_summary(clustering_result)
        
        # Journaliser le succès du stockage
        logger.info(f"Résultats du clustering sauvegardés dans {clustering_file}")
        
        # Pour la page de résultats, nous avons besoin d'une copie sécurisée pour l'affichage
        # Créer une version simplifiée (sérialisable) pour le rendu de la page
        display_result = {
            "success": clustering_result["success"],
            "algorithm": clustering_result["algorithm"],
            "n_clusters": clustering_result["n_clusters"],
            "columns_used": clustering_result["columns_used"],
            "labels": clustering_result["labels"],
            "cluster_sizes": clustering_result["cluster_sizes"],
            "pca_result": clustering_result["pca_result"],
            "pca_explained_variance": clustering_result["pca_explained_variance"]
        }
        
        # Ajouter des éléments spécifiques à l'algorithme
        if algorithm == 'kmeans':
            display_result["inertia"] = clustering_result.get("inertia")
            display_result["silhouette_score"] = clustering_result.get("silhouette_score")
            display_result["calinski_harabasz_score"] = clustering_result.get("calinski_harabasz_score")
            display_result["cluster_stats"] = clustering_result.get("cluster_stats", [])
        elif algorithm == 'dbscan':
            display_result["noise_points"] = clustering_result.get("noise_points")
            display_result["silhouette_score"] = clustering_result.get("silhouette_score")
            display_result["calinski_harabasz_score"] = clustering_result.get("calinski_harabasz_score")
            display_result["cluster_stats"] = clustering_result.get("cluster_stats", {})
        elif algorithm == 'hierarchical':
            display_result["silhouette_score"] = clustering_result.get("silhouette_score")
            display_result["calinski_harabasz_score"] = clustering_result.get("calinski_harabasz_score")
            display_result["cluster_stats"] = clustering_result.get("cluster_stats", [])
        
        # Remettre le DataFrame de résultats si nécessaire pour l'affichage
        if "result_df_copy" in locals():
            transformation_manager.save_transformed_dataframe(file_id, result_df_copy)
        
        return render_template('clustering.html',
                               df_info={'shape': df.shape},
                               numeric_columns=df.select_dtypes(include=['number']).columns.tolist(),
                               filename=filename,
                               clustering_result=display_result,
                               cluster_summary=cluster_summary)
    
    except Exception as e:
        app.logger.error(f"Erreur lors du clustering: {e}", exc_info=True)
        flash(f'Erreur lors du clustering: {str(e)}', 'danger')
        return redirect(url_for('clustering'))

@app.route('/save_clustering', methods=['POST'])
def save_clustering():
    """Sauvegarde les résultats du clustering dans le DataFrame"""
    # Vérifier que le fichier est bien chargé
    file_id = session.get('file_id')
    filename = session.get('filename')
    
    if not file_id or not filename:
        flash('Aucune donnée chargée. Veuillez d\'abord charger un fichier CSV ou accéder à la base de données.', 'warning')
        return redirect(url_for('data_processing'))
    
    # Récupérer le DataFrame courant
    df = transformation_manager.get_current_dataframe(file_id)
    
    if df is None:
        flash('Erreur lors de la récupération des données. Veuillez recharger le fichier.', 'danger')
        return redirect(url_for('data_processing'))
    
    # Récupérer les résultats du clustering de la session
    clustering_result = session.get('clustering_result')
    
    if not clustering_result:
        flash('Aucun résultat de clustering trouvé. Veuillez d\'abord exécuter le clustering.', 'warning')
        return redirect(url_for('clustering'))
    
    try:
        # Récupérer le nom de la colonne pour les clusters
        cluster_column_name = request.form.get('cluster_column_name', f"cluster_{clustering_result['algorithm']}")
        
        # Ajouter les clusters au DataFrame
        df_with_clusters = df.copy()
        
        # Créer une colonne pour les clusters
        df_with_clusters[cluster_column_name] = clustering_result['labels']
        
        # Sauvegarder le DataFrame mis à jour
        transformation_manager.save_transformed_dataframe(file_id, df_with_clusters)
        
        # Ajouter la transformation à l'historique
        transformation_manager.add_transformation(file_id, {
            "type": "clustering",
            "params": {
                "algorithm": clustering_result['algorithm'],
                "n_clusters": clustering_result['n_clusters'],
                "column_name": cluster_column_name
            },
            "timestamp": datetime.now().isoformat()
        })
        
        flash(f'Clusters sauvegardés dans la colonne "{cluster_column_name}"', 'success')
        return redirect(url_for('data_preview'))
    
    except Exception as e:
        flash(f'Erreur lors de la sauvegarde des clusters: {str(e)}', 'danger')
        return redirect(url_for('clustering'))

@app.route('/export_clusters', methods=['GET'])
def export_clusters():
    """Exporte les résultats du clustering au format spécifié"""
    # Vérifier que le fichier est bien chargé
    file_id = session.get('file_id')
    filename = session.get('filename')
    
    if not file_id or not filename:
        flash('Aucune donnée chargée. Veuillez d\'abord charger un fichier CSV ou accéder à la base de données.', 'warning')
        return redirect(url_for('data_processing'))
    
    # Récupérer le DataFrame courant
    df = transformation_manager.get_current_dataframe(file_id)
    
    if df is None:
        flash('Erreur lors de la récupération des données. Veuillez recharger le fichier.', 'danger')
        return redirect(url_for('data_processing'))
    
    # Récupérer les résultats du clustering de la session
    clustering_result = session.get('clustering_result')
    
    if not clustering_result:
        flash('Aucun résultat de clustering trouvé. Veuillez d\'abord exécuter le clustering.', 'warning')
        return redirect(url_for('clustering'))
    
    try:
        # Format d'exportation
        export_format = request.args.get('format', 'csv')
        
        # Ajouter les clusters au DataFrame
        df_with_clusters = df.copy()
        cluster_column_name = f"cluster_{clustering_result['algorithm']}"
        df_with_clusters[cluster_column_name] = clustering_result['labels']
        
        # Créer un fichier temporaire pour l'exportation
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{export_format}')
        
        if export_format == 'csv':
            df_with_clusters.to_csv(temp_file.name, index=False)
            mime_type = 'text/csv'
        elif export_format == 'xlsx':
            df_with_clusters.to_excel(temp_file.name, index=False)
            mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        else:
            temp_file.close()
            os.unlink(temp_file.name)
            flash(f'Format d\'exportation non pris en charge: {export_format}', 'danger')
            return redirect(url_for('clustering'))
        
        temp_file.close()
        
        # Renvoyer le fichier
        export_filename = f"{os.path.splitext(filename)[0]}_clusters.{export_format}"
        return send_file(temp_file.name, 
                         as_attachment=True, 
                         download_name=export_filename, 
                         mimetype=mime_type)
    
    except Exception as e:
        flash(f'Erreur lors de l\'exportation des clusters: {str(e)}', 'danger')
        return redirect(url_for('clustering'))

@app.route('/history')
def history():
    """Page d'historique des analyses"""
    try:
        # Récupérer les analyses CSV
        csv_analyses = history_manager.get_recent_analyses(20)
        
        # Récupérer les analyses PDF
        pdf_analyses = pdf_history_manager.get_recent_pdf_analyses(20)
        
        return render_template('history.html', csv_analyses=csv_analyses, pdf_analyses=pdf_analyses)
    
    except Exception as e:
        flash(f'Erreur lors de la récupération de l\'historique: {str(e)}', 'danger')
        return render_template('history.html', csv_analyses=[], pdf_analyses=[])

@app.route('/dashboard')
def dashboard():
    """Page du tableau de bord analytique"""
    # Vérifier que le fichier est bien chargé
    file_id = session.get('file_id')
    
    if not file_id:
        return render_template('dashboard.html')
    
    # Récupérer le DataFrame courant
    df = transformation_manager.get_current_dataframe(file_id)
    
    if df is None:
        return render_template('dashboard.html')
    
    # Générer un rapport pour le tableau de bord
    try:
        dashboard_data = generate_report(df)
        return render_template('dashboard.html', dashboard_data=dashboard_data)
    except Exception as e:
        flash(f'Erreur lors de la génération du tableau de bord: {str(e)}', 'danger')
        return render_template('dashboard.html')

@app.route('/pdf_analysis')
def pdf_analysis():
    """Page d'analyse de documents PDF"""
    return render_template('pdf_analysis.html')

@app.route('/settings')
def settings():
    """Page des paramètres de l'application"""
    return render_template('settings.html')


# API Routes
@app.route('/api/generate_visualization', methods=['POST'])
def api_generate_visualization():
    """API pour générer une visualisation"""
    # Vérifier si des données sont disponibles
    file_id = session.get('file_id')
    
    if not file_id:
        return jsonify({"error": "Aucune donnée disponible"})
    
    # Récupérer le DataFrame
    df = transformation_manager.get_current_dataframe(file_id)
    
    if df is None:
        return jsonify({"error": "Erreur lors de la récupération des données"})
    
    # Récupérer les paramètres de la requête
    data = request.json
    chart_type = data.get('chart_type')
    x_var = data.get('x_var')
    y_var = data.get('y_var')
    color_var = data.get('color_var')
    
    # Paramètres supplémentaires
    params = {k: v for k, v in data.items() if k not in ['chart_type', 'x_var', 'y_var', 'color_var']}
    
    try:
        # Utiliser la fonction du module de visualisation
        result = create_visualization(df, chart_type, x_var, y_var, color_var, **params)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/elbow_method', methods=['POST'])
def api_elbow_method():
    """API pour calculer les données de la méthode du coude"""
    # Vérifier si des données sont disponibles
    file_id = session.get('file_id')
    
    if not file_id:
        return jsonify({"error": "Aucune donnée disponible", "success": False})
    
    # Récupérer le DataFrame
    df = transformation_manager.get_current_dataframe(file_id)
    
    if df is None:
        return jsonify({"error": "Erreur lors de la récupération des données", "success": False})
    
    # Récupérer les colonnes sélectionnées
    data = request.json
    columns = data.get('columns', [])
    
    try:
        # Initialiser le processeur de clustering
        clustering_processor = ClusteringProcessor()
        
        # Calculer les données de la méthode du coude
        result = clustering_processor.get_elbow_method_data(df, columns)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "success": False})

@app.route('/api/column_values', methods=['POST'])
def api_column_values():
    """API pour récupérer les valeurs uniques d'une colonne"""
    # Vérifier si des données sont disponibles
    file_id = session.get('file_id')
    
    if not file_id:
        return jsonify({"error": "Aucune donnée disponible"})
    
    # Récupérer le DataFrame
    df = transformation_manager.get_current_dataframe(file_id)
    
    if df is None:
        return jsonify({"error": "Erreur lors de la récupération des données"})
    
    # Récupérer le nom de la colonne
    data = request.json
    column_name = data.get('column_name')
    
    if not column_name or column_name not in df.columns:
        return jsonify({"error": f"Colonne '{column_name}' non trouvée"})
    
    try:
        # Récupérer les valeurs uniques et leur fréquence
        value_counts = df[column_name].value_counts().reset_index()
        value_counts.columns = ['value', 'count']
        
        # Convertir en liste de dictionnaires
        values = value_counts.to_dict('records')
        
        # Statistiques sur la colonne
        stats = {
            "total_rows": len(df),
            "unique_values": df[column_name].nunique(),
            "missing_values": df[column_name].isna().sum(),
            "data_type": str(df[column_name].dtype)
        }
        
        return jsonify({"values": values, "stats": stats})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/analyze_clusters_with_ai', methods=['POST'])
def api_analyze_clusters_with_ai():
    """API pour analyser les clusters avec l'IA via Ollama"""
    import logging
    import traceback
    import pickle
    
    logger = logging.getLogger(__name__)
    logger.info("Début de l'analyse des clusters avec IA")
    
    # Récupérer l'ID du clustering
    clustering_id = session.get('clustering_id')
    
    if not clustering_id:
        logger.warning("Aucun ID de clustering trouvé dans la session")
        return jsonify({"success": False, "error": "Aucun résultat de clustering trouvé"}), 400
    
    # Chemin du fichier contenant les résultats du clustering
    # Corriger le chemin pour inclure le sous-dossier clustering_results
    clustering_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'clustering_results')
    clustering_file = os.path.join(clustering_folder, f"{clustering_id}.pkl")
    
    logger.info(f"Recherche du fichier de clustering: {clustering_file}")
    
    if not os.path.exists(clustering_file):
        logger.warning(f"Fichier de clustering non trouvé: {clustering_file}")
        return jsonify({"success": False, "error": "Résultats de clustering non disponibles"}), 400
    
    # Charger les résultats du clustering
    try:
        with open(clustering_file, 'rb') as f:
            clustering_result = pickle.load(f)
        
        logger.info(f"Fichier de clustering chargé avec succès: {clustering_id}")
    except Exception as e:
        logger.error(f"Erreur lors du chargement des résultats de clustering: {e}")
        return jsonify({"success": False, "error": f"Erreur lors du chargement des résultats: {str(e)}"}), 500
    
    # Récupérer le contexte utilisateur
    data = request.json or {}
    user_context = data.get('user_context', '')
    logger.info(f"Contexte utilisateur reçu: {user_context[:50]}...")
    
    try:
        # S'assurer que clustering_processor est disponible
        if 'clustering_processor' not in globals():
            from modules.clustering_module import ClusteringProcessor
            global clustering_processor
            clustering_processor = ClusteringProcessor()
            logger.info("Processeur de clustering initialisé")
        
        # Analyser les clusters
        logger.info("Démarrage de l'analyse des clusters avec IA...")
        analysis_result = clustering_processor.analyze_clusters_with_ai(clustering_result, user_context)
        
        if analysis_result.get("success", False):
            logger.info(f"Analyse réussie, {len(analysis_result.get('analysis', ''))} caractères générés")
            # Tronquer pour le log
            analysis_excerpt = analysis_result.get('analysis', '')[:100] + "..." if analysis_result.get('analysis') else ""
            logger.info(f"Extrait: {analysis_excerpt}")
        else:
            logger.warning(f"Échec de l'analyse: {analysis_result.get('error')}")
        
        return jsonify(analysis_result)
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Exception non gérée lors de l'analyse IA: {str(e)}\n{error_trace}")
        return jsonify({
            "success": False, 
            "error": str(e),
            "trace": error_trace
        }), 500

@app.route('/api/dataset_stats')
def api_dataset_stats():
    """API pour obtenir des statistiques sur le dataset"""
    # Vérifier si des données sont disponibles
    file_id = session.get('file_id')
    
    if not file_id:
        return jsonify({"error": "Aucune donnée disponible"})
    
    # Récupérer le DataFrame
    df = transformation_manager.get_current_dataframe(file_id)
    
    if df is None:
        return jsonify({"error": "Erreur lors de la récupération des données"})
    
    try:
        # Statistiques générales
        stats = {
            "rows": len(df),
            "columns": len(df.columns),
            "missing_values": int(df.isna().sum().sum()),
            "missing_percentage": float(df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
        }
        
        # Statistiques sur les colonnes numériques
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            stats["numeric_stats"] = {
                "count": len(numeric_cols),
                "columns": numeric_cols.tolist()
            }
        
        # Statistiques sur les colonnes catégorielles
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            stats["categorical_stats"] = {
                "count": len(cat_cols),
                "columns": cat_cols.tolist()
            }
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)})

# Fonctions utilitaires
def allowed_file(filename, allowed_extensions=None):
    """Vérifie si le fichier a une extension autorisée"""
    if allowed_extensions is None:
        allowed_extensions = app.config['ALLOWED_EXTENSIONS']
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions




@app.route('/pdf_analysis_results/<pdf_id>')
def pdf_analysis_results(pdf_id):
    """Page des résultats d'analyse PDF"""
    # Récupérer l'analyse PDF
    analysis = pdf_history_manager.get_pdf_analysis(pdf_id)
    
    if not analysis:
        flash('Analyse PDF non trouvée', 'warning')
        return redirect(url_for('pdf_analysis'))
    
    return render_template('pdf_analysis_results.html', analysis=analysis)


## --------------------------------- CALENDRIER ------------------------------------------------------------
# route pour la page du calendrier
@app.route('/api/calendar_data', methods=['POST'])
def api_calendar_data():
    """API pour obtenir les données du calendrier des achats avec filtres avancés"""
    try:
        # Récupérer les filtres depuis la requête
        data = request.json
        
        # Normaliser les valeurs des filtres
        brand_filter = data.get('brand', 'all')
        date_start = data.get('date_start', None)
        date_end = data.get('date_end', None)
        payment_method = data.get('payment_method', 'all')
        article_filter = data.get('article', 'all')
        
        # Normaliser le genre en minuscules pour éviter les problèmes de casse
        gender = data.get('gender', 'all')
        if gender != 'all':
            gender = gender.lower()
        
        age_range = data.get('age_range', 'all')
        
        # Log détaillé des filtres reçus
        logger.info(f"Filtres reçus: brand={brand_filter}, payment={payment_method}, article={article_filter}, gender={gender}, age={age_range}")
        logger.info(f"Requête complète: {data}")
        
        # Vérifier si on utilise un DataFrame chargé ou la base de données
        file_id = session.get('file_id')
        
        if not file_id:
            # Utiliser la base de données SQLite
            db_path = 'modules/fidelity_db.sqlite'
            
            # Vérifier que la base de données existe
            if not os.path.exists(db_path):
                logger.error(f"Base de données non trouvée: {db_path}")
                return jsonify({
                    'success': False,
                    'error': "Base de données non trouvée"
                })
            
            # Connexion à la base de données
            conn = sqlite3.connect(db_path)
            
            # Construction de la requête SQL de base
            query = """
            SELECT 
                date(t.date_transaction) as date_achat,
                COUNT(*) as nb_achats,
                m.nom as magasin,
                t.type_paiement as moyen_paiement,
                c.genre,
                CAST(strftime('%Y', 'now') - strftime('%Y', c.date_naissance) AS INTEGER) as age
            FROM transactions t
            JOIN points_vente m ON t.magasin_id = m.magasin_id
            JOIN clients c ON t.client_id = c.client_id
            """
            
            # Filtres article (avec jointures)
            if article_filter != 'all':
                query = query.replace("FROM transactions t", """
                FROM transactions t
                JOIN details_transactions dt ON t.transaction_id = dt.transaction_id
                JOIN produits p ON dt.produit_id = p.produit_id
                """)
            
            # Conditions WHERE
            conditions = []
            params = []
            
            # Filtre par magasin
            if brand_filter != 'all':
                conditions.append("m.nom = ?")
                params.append(brand_filter)
            
            # Filtre par moyen de paiement
            if payment_method != 'all':
                conditions.append("t.type_paiement = ?")
                params.append(payment_method)
            
            # Filtre par article
            if article_filter != 'all':
                conditions.append("p.nom LIKE ?")  # Recherche flexible avec LIKE
                params.append(f"%{article_filter}%")
                logger.info(f"Filtrage par article: {article_filter}")
            
            # Filtre par genre - insensible à la casse
            if gender != 'all':
                conditions.append("LOWER(c.genre) = ?")  # Conversion en minuscules pour comparaison insensible à la casse
                params.append(gender.lower())
                logger.info(f"Filtrage par genre: {gender}")
            
            # Filtre par tranche d'âge
            if age_range != 'all':
                logger.info(f"Filtrage par tranche d'âge: {age_range}")
                if age_range == "0-18":
                    conditions.append("(strftime('%Y', 'now') - strftime('%Y', c.date_naissance)) < 19")
                elif age_range == "19-25":
                    conditions.append("(strftime('%Y', 'now') - strftime('%Y', c.date_naissance)) BETWEEN 19 AND 25")
                elif age_range == "26-35":
                    conditions.append("(strftime('%Y', 'now') - strftime('%Y', c.date_naissance)) BETWEEN 26 AND 35")
                elif age_range == "36-50":
                    conditions.append("(strftime('%Y', 'now') - strftime('%Y', c.date_naissance)) BETWEEN 36 AND 50")
                elif age_range == "51+":
                    conditions.append("(strftime('%Y', 'now') - strftime('%Y', c.date_naissance)) > 50")
            
            # Filtre par période
            if date_start:
                conditions.append("date(t.date_transaction) >= ?")
                params.append(date_start)
            if date_end:
                conditions.append("date(t.date_transaction) <= ?")
                params.append(date_end)
            
            # Ajouter les conditions à la requête
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            # Grouper par date, magasin et moyen de paiement (sans inclure genre et âge)
            query += " GROUP BY date_achat, magasin, moyen_paiement"
            
            # Exécuter la requête
            logger.info(f"Requête SQL: {query}")
            logger.info(f"Paramètres: {params}")
            
            df = pd.read_sql_query(query, conn, params=params)
            
            # Récupérer les listes de valeurs pour les filtres
            # Magasins
            brands_query = "SELECT DISTINCT nom FROM points_vente ORDER BY nom"
            brands_df = pd.read_sql_query(brands_query, conn)
            brands = brands_df['nom'].tolist()
            
            # Moyens de paiement
            payment_query = "SELECT DISTINCT type_paiement FROM transactions WHERE type_paiement IS NOT NULL ORDER BY type_paiement"
            payment_df = pd.read_sql_query(payment_query, conn)
            payment_methods = payment_df['type_paiement'].tolist()
            
            # Articles (produits)
            products_query = """
            SELECT p.nom, COUNT(*) as count
            FROM details_transactions dt
            JOIN produits p ON dt.produit_id = p.produit_id
            GROUP BY p.nom
            ORDER BY count DESC
            LIMIT 30
            """
            products_df = pd.read_sql_query(products_query, conn)
            articles = products_df['nom'].tolist()
            
            # Genres disponibles
            genders_query = "SELECT DISTINCT genre FROM clients WHERE genre IS NOT NULL"
            genders_df = pd.read_sql_query(genders_query, conn)
            available_genders = genders_df['genre'].tolist()
            
            conn.close()
            
            # Vérifier que des données ont été récupérées
            if df.empty:
                return jsonify({
                    'success': True,
                    'dates': [],
                    'values': [],
                    'brands': brands,
                    'paymentInfo': [],
                    'magasins': [],
                    'payment_methods': payment_methods,
                    'articles': articles,
                    'genders': ['homme', 'femme'],
                    'age_ranges': ['0-18', '19-25', '26-35', '36-50', '51+']
                })
                
            # Préparer les données détaillées
            date_col = 'date_achat'
            value_col = 'nb_achats'
            payment_col = 'moyen_paiement'
            store_col = 'magasin'
            
        else:
            # Utiliser le DataFrame chargé en session
            df_original = transformation_manager.get_current_dataframe(file_id)
            # Faire une copie pour ne pas modifier l'original
            df = df_original.copy()
            
            # Log détaillé des colonnes disponibles
            logger.info(f"Colonnes disponibles dans le DataFrame: {df.columns.tolist()}")
            
            # Vérification de présence des colonnes de genre et âge
            if 'genre' in df.columns:
                unique_genres = df['genre'].dropna().unique()
                logger.info(f"Valeurs uniques de genre: {unique_genres}")
            else:
                logger.warning("Colonne 'genre' non trouvée dans le DataFrame")
                
            if 'age' in df.columns:
                logger.info(f"Statistiques d'âge: min={df['age'].min()}, max={df['age'].max()}, moyenne={df['age'].mean()}")
            else:
                logger.warning("Colonne 'age' non trouvée dans le DataFrame")
            
            # Déterminer les colonnes à utiliser en fonction de ce qui est disponible
            # Colonnes de date possibles
            date_columns = ['date_transaction', 'date_achat', 'date', 'date_operation']
            date_col = None
            for col in date_columns:
                if col in df.columns:
                    date_col = col
                    break
            
            if not date_col:
                return jsonify({
                    'success': False,
                    'error': f"Aucune colonne de date trouvée. Colonnes disponibles: {df.columns.tolist()}"
                })
            
            # Convertir la colonne de date en datetime
            try:
                df[date_col] = pd.to_datetime(df[date_col])
            except Exception as e:
                logger.error(f"Erreur de conversion de la date: {e}")
                try:
                    df[date_col] = pd.to_datetime(df[date_col].astype(str))
                except:
                    return jsonify({
                        'success': False,
                        'error': f"Impossible de convertir la colonne {date_col} en date"
                    })
            
            # Filtre par période
            if date_start:
                df = df[df[date_col] >= pd.to_datetime(date_start)]
            if date_end:
                df = df[df[date_col] <= pd.to_datetime(date_end)]
            
            # Appliquer la conversion à date (sans heure)
            df[date_col] = df[date_col].dt.date
            
            # Identifier les colonnes possibles
            store_columns = ['magasin', 'nom_magasin', 'enseigne', 'email']
            payment_columns = ['moyen_paiement', 'type_paiement', 'payment_method']
            gender_columns = ['genre', 'gender', 'sexe']
            age_columns = ['age', 'age_client']
            article_columns = ['nom_article', 'article', 'produit', 'nom_produit', 'nom']
            
            # Trouver les colonnes disponibles
            store_col = None
            for col in store_columns:
                if col in df.columns:
                    store_col = col
                    break
            
            payment_col = None  
            for col in payment_columns:
                if col in df.columns:
                    payment_col = col
                    break
            
            gender_col = None
            for col in gender_columns:
                if col in df.columns:
                    gender_col = col
                    break
            
            age_col = None
            for col in age_columns:
                if col in df.columns:
                    age_col = col
                    break
            
            article_col = None
            for col in article_columns:
                if col in df.columns:
                    article_col = col
                    break
            
            logger.info(f"Colonnes identifiées: date={date_col}, store={store_col}, payment={payment_col}, gender={gender_col}, age={age_col}, article={article_col}")
            
            # Appliquer les filtres
            # Par magasin
            if brand_filter != 'all' and store_col in df.columns:
                df = df[df[store_col] == brand_filter]
                logger.info(f"Après filtre magasin: {len(df)} lignes")
            
            # Par moyen de paiement
            if payment_method != 'all' and payment_col in df.columns:
                df = df[df[payment_col] == payment_method]
                logger.info(f"Après filtre paiement: {len(df)} lignes")
            
            # Par article
            if article_filter != 'all' and article_col in df.columns:
                df = df[df[article_col].str.contains(article_filter, case=False, na=False)]
                logger.info(f"Après filtre article: {len(df)} lignes")
            
            # Par genre - avec plusieurs techniques de correspondance
            if gender != 'all' and gender_col in df.columns:
                # Approche 1: Correspondance exacte
                mask1 = df[gender_col] == gender
                # Approche 2: Correspondance insensible à la casse
                mask2 = df[gender_col].str.lower() == gender.lower()
                # Approche 3: Correspondance partielle
                mask3 = df[gender_col].str.contains(gender, case=False, na=False)
                
                # Combiner les approches avec OU logique
                combined_mask = mask1 | mask2 | mask3
                
                # Filtrer
                df_filtered = df[combined_mask]
                
                if not df_filtered.empty:
                    df = df_filtered
                    logger.info(f"Filtrage genre réussi: {len(df)} lignes")
                else:
                    logger.warning(f"Aucune correspondance pour le genre '{gender}'. Valeurs disponibles: {df[gender_col].unique()}")
            
            # Par tranche d'âge
            if age_range != 'all' and age_col in df.columns:
                # Essayer de convertir en numérique s'il ne l'est pas déjà
                if not pd.api.types.is_numeric_dtype(df[age_col]):
                    try:
                        df[age_col] = pd.to_numeric(df[age_col], errors='coerce')
                        logger.info(f"Colonne d'âge '{age_col}' convertie en numérique")
                    except Exception as e:
                        logger.error(f"Impossible de convertir la colonne d'âge en numérique: {e}")
                
                # Appliquer le filtre d'âge
                original_count = len(df)
                
                if age_range == "0-18":
                    df = df[df[age_col] < 19]
                elif age_range == "19-25":
                    df = df[(df[age_col] >= 19) & (df[age_col] <= 25)]
                elif age_range == "26-35":
                    df = df[(df[age_col] >= 26) & (df[age_col] <= 35)]
                elif age_range == "36-50":
                    df = df[(df[age_col] >= 36) & (df[age_col] <= 50)]
                elif age_range == "51+":
                    df = df[df[age_col] > 50]
                
                logger.info(f"Filtre âge '{age_range}' appliqué: {len(df)}/{original_count} lignes")
            
            # Vérifier si on a encore des données après filtrage
            if df.empty:
                logger.warning("DataFrame vide après application des filtres")
                
                # Récupérer les articles disponibles du DataFrame original pour le menu déroulant
                articles = []
                if article_col and article_col in df_original.columns:
                    articles = df_original[article_col].dropna().unique().tolist()[:30]
                
                # Récupérer les genres disponibles
                available_genders = []
                if gender_col and gender_col in df_original.columns:
                    available_genders = df_original[gender_col].dropna().unique().tolist()
                else:
                    available_genders = ['homme', 'femme']
                
                return jsonify({
                    'success': True,
                    'dates': [],
                    'values': [],
                    'brands': [brand_filter] if brand_filter != 'all' else ['Tous les magasins'],
                    'paymentInfo': [],
                    'magasins': [],
                    'payment_methods': [payment_method] if payment_method != 'all' else [],
                    'articles': articles,
                    'genders': available_genders,
                    'age_ranges': ['0-18', '19-25', '26-35', '36-50', '51+']
                })
            
            # Grouper par les colonnes disponibles - EXCLURE genre et âge pour permettre le filtrage
            group_columns = [date_col]
            if store_col:
                group_columns.append(store_col)
            if payment_col:
                group_columns.append(payment_col)
            
            # Effectuer le groupement
            try:
                grouped = df.groupby(group_columns).size().reset_index(name='nb_achats')
                df = grouped
                value_col = 'nb_achats'
                logger.info(f"Groupement réussi: {len(df)} lignes de résultat")
            except Exception as e:
                logger.error(f"Erreur lors du groupement: {e}")
                return jsonify({
                    'success': False,
                    'error': f"Erreur lors du groupement des données: {e}"
                })
            
            # Récupérer les valeurs possibles pour les filtres
            brands = df[store_col].unique().tolist() if store_col in df.columns else ['Tous les magasins']
            payment_methods = df[payment_col].unique().tolist() if payment_col in df.columns else []
            
            # Pour les articles, récupérer depuis le DF original pour avoir toutes les options
            articles = []
            if article_col and article_col in df_original.columns:
                articles = df_original[article_col].dropna().unique().tolist()[:30]
            
            # Pour le genre, utiliser les valeurs du DataFrame original
            available_genders = []
            if gender_col and gender_col in df_original.columns:
                available_genders = df_original[gender_col].dropna().unique().tolist()
            else:
                available_genders = ['homme', 'femme']
        
        # Préparer les données pour la réponse JSON
        # Convertir les dates en strings
        if len(df) > 0:  # Vérifier que df n'est pas vide
            if hasattr(df[date_col].iloc[0], 'strftime'):
                dates = [d.strftime('%Y-%m-%d') for d in df[date_col]]
            else:
                dates = [str(d) for d in df[date_col]]
            
            values = df[value_col].tolist()
            
            # Informations supplémentaires si disponibles
            paymentInfo = df[payment_col].tolist() if payment_col in df.columns else [None] * len(dates)
            magasins = df[store_col].tolist() if store_col in df.columns else [None] * len(dates)
            
            # On ne renvoie pas les genres et âges dans ces listes car on a filtré avant
            genders_data = [gender] * len(dates) if gender != 'all' else [None] * len(dates)
            ages_data = [age_range] * len(dates) if age_range != 'all' else [None] * len(dates)
        else:
            dates = []
            values = []
            paymentInfo = []
            magasins = []
            genders_data = []
            ages_data = []
        
        # Ajouter les tranches d'âge standardisées
        age_ranges = ['0-18', '19-25', '26-35', '36-50', '51+']
        
        return jsonify({
            'success': True,
            'dates': dates,
            'values': values,
            'brands': brands,
            'paymentInfo': paymentInfo,
            'magasins': magasins,
            'genders': genders_data,
            'ages': ages_data,
            'payment_methods': payment_methods,
            'articles': articles,
            'available_genders': available_genders,
            'available_age_ranges': age_ranges
        })
    
    except Exception as e:
        logger.exception(f"Erreur lors de la récupération des données du calendrier: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/calendar')  # Changez le décorateur si nécessaire
def calendar_view():
    """Page de visualisation des achats en calendrier"""
    # Vérifiez que le fichier est bien chargé
    file_id = session.get('file_id')
    
    if not file_id:
        flash('Aucune donnée chargée. Veuillez d\'abord charger un fichier CSV ou accéder à la base de données.', 'warning')
        return render_template('calendar_view.html', has_data=False)
    
    # Récupérer le DataFrame
    df = transformation_manager.get_current_dataframe(file_id)
    
    if df is None:
        flash('Erreur lors de la récupération des données. Veuillez recharger le fichier.', 'warning')
        return render_template('calendar_view.html', has_data=False)
    
    return render_template('calendar_view.html', 
                           filename=session.get('filename'),
                           has_data=True)

# ---------------------------------------------------------CLIENT ---------------------------------------------------

@app.route('/api/client_demographics')
def api_client_demographics():
    """API pour récupérer les données démographiques des clients"""
    # Vérifier si des données sont disponibles
    file_id = session.get('file_id')
    
    if not file_id:
        return jsonify({"success": False, "error": "Aucune donnée disponible"})
    
    # Récupérer le DataFrame
    df = transformation_manager.get_current_dataframe(file_id)
    
    if df is None:
        return jsonify({"success": False, "error": "Erreur lors de la récupération des données"})
    
    try:
        demographics_data = {}
        
        # Distribution par genre
        if 'genre' in df.columns:
            gender_counts = df['genre'].value_counts().to_dict()
            # Normaliser les clés en minuscules pour cohérence
            normalized_gender_counts = {}
            for key, value in gender_counts.items():
                if pd.notna(key):
                    normalized_key = key.lower() if isinstance(key, str) else key
                    normalized_gender_counts[normalized_key] = value
            demographics_data['gender_distribution'] = normalized_gender_counts
        
        # Distribution par âge
        if 'age' in df.columns:
            # Définir les tranches d'âge
            age_bins = [0, 18, 25, 35, 50, 100]
            age_labels = ['0-18', '19-25', '26-35', '36-50', '51+']
            
            # Créer une copie pour éviter les avertissements de modification avec pd.cut
            df_age = df.copy()
            
            # Convertir en numérique si pas déjà fait
            if not pd.api.types.is_numeric_dtype(df_age['age']):
                df_age['age'] = pd.to_numeric(df_age['age'], errors='coerce')
            
            # Catégoriser les âges
            df_age['age_group'] = pd.cut(df_age['age'], bins=age_bins, labels=age_labels, right=False)
            
            # Compter les occurrences par tranche d'âge
            age_counts = df_age['age_group'].value_counts().sort_index()
            
            # Formater pour le graphique
            demographics_data['age_distribution'] = {
                'categories': age_counts.index.tolist(),
                'values': age_counts.values.tolist()
            }
        
        # Distribution par segment client
        if 'segment_client' in df.columns:
            segment_counts = df['segment_client'].value_counts().to_dict()
            demographics_data['segment_distribution'] = segment_counts
            
            # Panier moyen par segment
            if 'montant_total' in df.columns:
                avg_basket = df.groupby('segment_client')['montant_total'].mean().round(2)
                
                demographics_data['avg_basket_by_segment'] = {
                    'categories': avg_basket.index.tolist(),
                    'values': avg_basket.values.tolist()
                }
        
        return jsonify({"success": True, **demographics_data})
        
    except Exception as e:
        app.logger.error(f"Erreur lors de la génération des statistiques démographiques: {e}")
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)




