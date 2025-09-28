#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aplicação Flask para Predição de Dengue
========================================

Esta aplicação web utiliza um modelo de machine learning pré-treinado
para fazer predições de dengue com base em dados clínicos inseridos
pelo usuário através de um formulário web.

Autor: Desenvolvedor Full-Stack Sênior
Data: Setembro 2025
"""

# Importações necessárias
from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import numpy as np
import os

# Inicialização da aplicação Flask
app = Flask(__name__)

# Variável global para armazenar o modelo carregado
model = None

def load_model():
    """
    Carrega o modelo de machine learning pré-treinado.

    Returns:
        object: Modelo carregado ou None se houver erro
    """
    global model
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'rf_model_dengue.pkl')
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        print("✅ Modelo carregado com sucesso!")
        return model
    except Exception as e:
        print(f"❌ Erro ao carregar o modelo: {str(e)}")
        return None

def prepare_features(data):
    """
    Prepara os dados recebidos do frontend para o formato esperado pelo modelo.

    Args:
        data (dict): Dados recebidos do formulário

    Returns:
        pd.DataFrame: DataFrame com as features na ordem correta
    """
    # Ordem exata das features esperada pelo modelo
    features_order = [
        'IDADE', 'CS_SEXO_F', 'CS_SEXO_I', 'CS_SEXO_M',
        'CS_RACA_AMARELA', 'CS_RACA_BRANCA', 'CS_RACA_IGNORADO',
        'CS_RACA_INDÍGENA', 'CS_RACA_PARDA', 'CS_RACA_PRETA',
        'FEBRE_NÃO', 'FEBRE_SIM', 'MIALGIA_NÃO', 'MIALGIA_SIM',
        'CEFALEIA_NÃO', 'CEFALEIA_SIM', 'EXANTEMA_NÃO', 'EXANTEMA_SIM',
        'VOMITO_NÃO', 'VOMITO_SIM', 'PETEQUIA_N_NÃO', 'PETEQUIA_N_SIM',
        'DIABETES_NÃO', 'DIABETES_SIM', 'HEMATOLOG_NÃO', 'HEMATOLOG_SIM',
        'HEPATOPAT_NÃO', 'HEPATOPAT_SIM', 'RENAL_NÃO', 'RENAL_SIM'
    ]
    
    # Cria um dicionário com todas as features zeradas
    processed_data = {feature: 0 for feature in features_order}
    
    # Atualiza o dicionário com os dados recebidos
    processed_data.update(data)
    
    # Cria o DataFrame na ordem correta
    df = pd.DataFrame([processed_data])[features_order]
    return df

# Carrega o modelo uma vez na inicialização
load_model()

@app.route('/')
def home():
    """
    Rota principal que renderiza a página HTML.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Rota para realizar a predição com base nos dados do formulário.
    """
    try:
        data = request.get_json(force=True)
        
        # Prepara os dados para o modelo
        df = prepare_features(data)
        
        # Realiza a predição
        prediction = model.predict(df)
        prediction_result = int(prediction[0])
        
        # Tenta obter as probabilidades (opcional)
        try:
            probabilities = model.predict_proba(df)
            prob_negative = float(probabilities[0][0])
            prob_positive = float(probabilities[0][1])
        except:
            prob_negative = None
            prob_positive = None
        
        # Retornar resultado em formato JSON
        response = {
            'prediction': prediction_result,
            'message': 'Positivo para Dengue' if prediction_result == 1 else 'Negativo para Dengue',
            'probability_negative': prob_negative,
            'probability_positive': prob_positive
        }
        
        return jsonify(response)
        
    except Exception as e:
        # Tratamento de erros
        return jsonify({
            'error': f'Erro interno do servidor: {str(e)}',
            'prediction': None
        }), 500

@app.route('/health')
def health_check():
    """
    Rota para verificar o status da aplicação.

    Returns:
        json: Status da aplicação
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'message': 'Aplicação funcionando corretamente'
    })