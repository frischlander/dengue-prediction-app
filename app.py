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
    # Lista das features na ordem exata esperada pelo modelo
    features = [
        'IDADE', 'CS_SEXO_F', 'CS_SEXO_I', 'CS_SEXO_M',
        'CS_RACA_AMARELA', 'CS_RACA_BRANCA', 'CS_RACA_IGNORADO',
        'CS_RACA_INDÍGENA', 'CS_RACA_PARDA', 'CS_RACA_PRETA',
        'FEBRE_NÃO', 'FEBRE_SIM', 'MIALGIA_NÃO', 'MIALGIA_SIM',
        'CEFALEIA_NÃO', 'CEFALEIA_SIM', 'EXANTEMA_NÃO', 'EXANTEMA_SIM',
        'VOMITO_NÃO', 'VOMITO_SIM', 'PETEQUIA_N_NÃO', 'PETEQUIA_N_SIM',
        'DIABETES_NÃO', 'DIABETES_SIM', 'HEMATOLOG_NÃO', 'HEMATOLOG_SIM',
        'HEPATOPAT_NÃO', 'HEPATOPAT_SIM', 'RENAL_NÃO', 'RENAL_SIM'
    ]
    
    # Criar DataFrame com as features na ordem correta
    df = pd.DataFrame([data], columns=features)
    
    return df

@app.route('/')
def index():
    """
    Rota principal que renderiza a página inicial com o formulário.
    
    Returns:
        str: Template HTML renderizado
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Rota para fazer predições com base nos dados enviados pelo frontend.
    
    Returns:
        json: Resultado da predição em formato JSON
    """
    try:
        # Verificar se o modelo foi carregado
        if model is None:
            return jsonify({
                'error': 'Modelo não carregado. Verifique se o arquivo rf_model_dengue.pkl existe.',
                'prediction': None
            }), 500
        
        # Obter dados JSON da requisição
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'Nenhum dado recebido.',
                'prediction': None
            }), 400
        
        # Preparar os dados para o modelo
        df = prepare_features(data)
        
        # Fazer a predição
        prediction = model.predict(df)
        
        # Extrair o resultado da predição (primeiro elemento do array)
        prediction_result = int(prediction[0])
        
        # Obter probabilidades se disponível
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

# Carregamento do modelo na inicialização
if __name__ == '__main__':
    print("🚀 Iniciando aplicação Flask para Predição de Dengue...")
    
    # Carregar o modelo
    load_model()
    
    if model is None:
        print("⚠️  AVISO: Aplicação iniciada sem modelo carregado!")
    
    # Executar a aplicação
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )
