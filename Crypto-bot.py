import asyncio
import json
import statistics
import math
import random
from datetime import datetime
from telegram import Update
from telegram.ext import Application, CommandHandler, CallbackContext
import aiohttp

# توکن ربات شما
TOKEN = "7902681382:AAF8jQK-9fDhyMB_6qPw4pGHcsxtDYolUH8"

# ==================== موتورهای تحلیل (اصلاح شده) ====================
class AdvancedAIEngine:
    """Next-Generation AI Analysis Engine (Fixed Syntax)"""
    
    @staticmethod
    def lstm_price_prediction(prices, lookback=60, forecast_days=7):
        """Advanced LSTM Neural Network for price prediction (Fixed Parentheses)"""
        if len(prices) < lookback + 30:
            return {"predictions": [], "confidence": 0, "model_performance": 0}
        
        recent_data = prices[-lookback:]
        
        # LSTM Feature Engineering
        features = []
        for i in range(10, len(recent_data)):
            window = recent_data[i-10:i]
            features.append({
                'price_momentum': (window[-1] - window[0]) / window[0],
                'volatility': statistics.stdev(window) / statistics.mean(window),
                'trend_strength': abs(sum([(window[j] - window[j-1]) for j in range(1, len(window))])),
                'acceleration': (window[-1] - window[-5]) - (window[-5] - window[-10])
            })
        
        # LSTM Memory Simulation
        cell_state = 0.5
        hidden_state = 0.3
        predictions = []
        current_price = prices[-1]
        
        for day in range(forecast_days):
            # LSTM Gates Simulation (Fixed Parentheses Here)
            forget_gate = 1 / (1 + math.exp(-((day + 1) * 0.1)))
            recent_volatility = statistics.stdev(prices[-20:]) / statistics.mean(prices[-20:])
            input_gate = 1 / (1 + math.exp(-(recent_volatility * 2)))
            momentum = (prices[-1] - prices[-10]) / prices[-10]
            output_gate = 1 / (1 + math.exp(-(momentum * 3)))
            
            # Cell State Update
            new_info = math.tanh(momentum + recent_volatility * 0.5)
            cell_state = forget_gate * cell_state + input_gate * new_info
            hidden_state = output_gate * math.tanh(cell_state)
            
            # Price Prediction
            price_change = hidden_state * 0.03
            predicted_price = current_price * (1 + price_change)
            
            predictions.append({
                'day': day + 1,
                'price': predicted_price,
                'confidence': abs(hidden_state) * 100,
                'cell_state': cell_state,
                'hidden_state': hidden_state
            })
            current_price = predicted_price
        
        # Model Performance
        if predictions:
            recent_predictions = [p['price'] for p in predictions[:3]]
            actual_trend = (prices[-1] - prices[-7]) / prices[-7] if prices[-7] else 0
            predicted_trend = (recent_predictions[-1] - prices[-1]) / prices[-1] if prices[-1] else 0
            performance = max(0, 100 - abs(actual_trend - predicted_trend) * 100)
        else:
            performance = 0
        
        return {
            "predictions": predictions,
            "confidence": statistics.mean([p['confidence'] for p in predictions]) if predictions else 0,
            "model_performance": performance,
            "lstm_states": {
                "final_cell_state": cell_state,
                "final_hidden_state": hidden_state
            }
        }
    
    @staticmethod
    def ensemble_prediction(prices, volumes):
        """Advanced Ensemble Model (Optimized)"""
        if len(prices) < 100:
            return {"ensemble_signal": "HOLD", "confidence": 0, "models": []}
        
        models_results = []
        
        # Gradient Boosting Simulator
        gb_features = []
        for i in range(20, len(prices)):
            window = prices[i-20:i]
            vol_window = volumes[i-20:i] if len(volumes) >= i else [1] * 20
            
            gb_features.append({
                'price_ma_ratio': window[-1] / (sum(window) / len(window)),
                'volume_trend': (sum(vol_window[-5:]) / 5) / (sum(vol_window[:5]) / 5),
                'price_acceleration': window[-1] - 2*window[-10] + window[-20],
                'volatility_change': statistics.stdev(window[-10:]) - statistics.stdev(window[:10])
            })
        
        gb_score = 0
        for feature in gb_features[-10:]:
            gb_score += feature['price_ma_ratio'] * 0.4
            gb_score += math.tanh(feature['volume_trend']) * 0.3
            gb_score += math.tanh(feature['price_acceleration'] / prices[-1]) * 0.3
        
        gb_signal = "BUY" if gb_score > 0.1 else "SELL" if gb_score < -0.1 else "HOLD"
        models_results.append({"name": "Gradient Boosting", "signal": gb_signal, "score": gb_score})
        
        # Random Forest Simulator
        rf_trees = []
        for tree in range(10):
            tree_score = 0
            if tree % 3 == 0:
                tree_score = (prices[-1] - prices[-5]) / prices[-5] if prices[-5] else 0
            elif tree % 3 == 1:
                if len(volumes) >= 10:
                    vol_ratio = volumes[-1] / (sum(volumes[-10:]) / 10)
                    tree_score = math.tanh(vol_ratio - 1)
            else:
                ma5 = sum(prices[-5:]) / 5
                ma20 = sum(prices[-20:]) / 20
                tree_score = (ma5 - ma20) / ma20 if ma20 else 0
            rf_trees.append(tree_score)
        
        rf_average = statistics.mean(rf_trees) if rf_trees else 0
        rf_signal = "BUY" if rf_average > 0.02 else "SELL" if rf_average < -0.02 else "HOLD"
        models_results.append({"name": "Random Forest", "signal": rf_signal, "score": rf_average})
        
        # Support Vector Machine Simulator
        svm_features = []
        for i in range(30, len(prices)):
            returns = [(prices[j] - prices[j-1]) / prices[j-1] for j in range(i-29, i) if prices[j-1]]
            svm_features.append({
                'return_mean': statistics.mean(returns) if returns else 0,
                'return_std': statistics.stdev(returns) if len(returns) > 1 else 0,
                'momentum': sum(returns[-5:]) if len(returns) >= 5 else 0,
                'trend_consistency': len([r for r in returns[-10:] if r > 0]) / 10 if len(returns) >= 10 else 0.5
            })
        
        if svm_features:
            latest_features = svm_features[-1]
            svm_distance = (latest_features['momentum'] * 2 + 
                           latest_features['trend_consistency'] - 0.5 -
                           latest_features['return_std'] * 5)
        else:
            svm_distance = 0
        
        svm_signal = "BUY" if svm_distance > 0.1 else "SELL" if svm_distance < -0.1 else "HOLD"
        models_results.append({"name": "SVM", "signal": svm_signal, "score": svm_distance})
        
        # Ensemble Voting
        buy_votes = sum(1 for model in models_results if model["signal"] == "BUY")
        sell_votes = sum(1 for model in models_results if model["signal"] == "SELL")
        hold_votes = sum(1 for model in models_results if model["signal"] == "HOLD")
        
        total_models = len(models_results)
        if buy_votes > sell_votes and buy_votes > hold_votes:
            ensemble_signal = "BUY"
            confidence = (buy_votes / total_models) * 100
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            ensemble_signal = "SELL"
            confidence = (sell_votes / total_models) * 100
        else:
            ensemble_signal = "HOLD"
            confidence = (hold_votes / total_models) * 100
        
        return {
            "ensemble_signal": ensemble_signal,
            "confidence": confidence,
            "models": models_results,
            "voting": {"BUY": buy_votes, "SELL": sell_votes, "HOLD": hold_votes}
        }
    
    @staticmethod
    def pattern_recognition_ai(prices):
        """Enhanced AI Pattern Recognition (Optimized)"""
        if len(prices) < 50:
            return {"patterns": [], "signals": []}
        
        patterns_found = []
        signals = []
        recent_prices = prices[-50:]
        
        # Peak and Valley Detection
        peaks = []
        valleys = []
        
        for i in range(2, len(recent_prices) - 2):
            if (recent_prices[i] > recent_prices[i-1] and 
                recent_prices[i] > recent_prices[i+1] and
                recent_prices[i] > recent_prices[i-2] and 
                recent_prices[i] > recent_prices[i+2]):
                peaks.append((i, recent_prices[i]))
            elif (recent_prices[i] < recent_prices[i-1] and 
                  recent_prices[i] < recent_prices[i+1] and
                  recent_prices[i] < recent_prices[i-2] and 
                  recent_prices[i] < recent_prices[i+2]):
                valleys.append((i, recent_prices[i]))
        
        # Advanced Pattern Detection
        if len(peaks) >= 2:
            last_two_peaks = peaks[-2:]
            if last_two_peaks[0][1] and abs(last_two_peaks[0][1] - last_two_peaks[1][1]) / last_two_peaks[0][1] < 0.02:
                patterns_found.append("Double Top")
                signals.append(("SELL", "AI detected double top reversal pattern", 0.8))
        
        if len(valleys) >= 2:
            last_two_valleys = valleys[-2:]
            if last_two_valleys[0][1] and abs(last_two_valleys[0][1] - last_two_valleys[1][1]) / last_two_valleys[0][1] < 0.02:
                patterns_found.append("Double Bottom")
                signals.append(("BUY", "AI detected double bottom reversal pattern", 0.8))
        
        # Head and Shoulders
        if len(peaks) >= 3:
            last_three = peaks[-3:]
            left_shoulder, head, right_shoulder = last_three
            if head[1] and left_shoulder[1] and right_shoulder[1]:
                if (head[1] > left_shoulder[1] and 
                    head[1] > right_shoulder[1] and
                    abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1] < 0.03):
                    patterns_found.append("Head and Shoulders")
                    signals.append(("SELL", "AI detected head and shoulders top pattern", 0.85))
        
        return {
            "patterns": patterns_found,
            "signals": signals,
            "peaks": len(peaks),
            "valleys": len(valleys)
        }

class AdvancedQuantEngine:
    """Advanced Quantitative Analysis Engine (Optimized)"""
    
    @staticmethod
    def monte_carlo_simulation(prices, days=30, simulations=1000):
        """Advanced Monte Carlo Simulation (Safe Division)"""
        if len(prices) < 30:
            return {"simulations": [], "statistics": {}, "risk_metrics": {}}
        
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] != 0:  # Avoid division by zero
                returns.append((prices[i] - prices[i-1]) / prices[i-1])
        
        if not returns:
            return {"simulations": [], "statistics": {}, "risk_metrics": {}}
        
        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns) if len(returns) > 1 else 0
        current_price = prices[-1]
        simulation_results = []
        
        for sim in range(simulations):
            sim_prices = [current_price]
            
            for day in range(days):
                random_factor = random.gauss(0, 1)
                daily_return = mean_return + std_return * random_factor
                
                # Mean reversion
                if sim_prices[-1] and current_price:
                    mean_reversion = -0.1 * (sim_prices[-1] - current_price) / current_price
                else:
                    mean_reversion = 0
                adjusted_return = daily_return + mean_reversion
                
                new_price = sim_prices[-1] * (1 + adjusted_return)
                sim_prices.append(new_price)
            
            simulation_results.append(sim_prices)
        
        # Statistics
        final_prices = [sim[-1] for sim in simulation_results]
        final_returns = []
        for fp in final_prices:
            if current_price != 0:
                final_returns.append((fp - current_price) / current_price)
        
        if not final_returns:
            return {"simulations": [], "statistics": {}, "risk_metrics": {}}
        
        statistics_dict = {
            "mean_final_price": statistics.mean(final_prices),
            "median_final_price": statistics.median(final_prices),
            "std_final_price": statistics.stdev(final_prices) if len(final_prices) > 1 else 0,
            "min_final_price": min(final_prices) if final_prices else 0,
            "max_final_price": max(final_prices) if final_prices else 0,
            "mean_return": statistics.mean(final_returns),
            "std_return": statistics.stdev(final_returns) if len(final_returns) > 1 else 0
        }
        
        # Risk Metrics
        sorted_returns = sorted(final_returns)
        var_95 = sorted_returns[int(0.05 * len(sorted_returns))] if sorted_returns else 0
        var_99 = sorted_returns[int(0.01 * len(sorted_returns))] if sorted_returns else 0
        
        upside_scenarios = len([r for r in final_returns if r > 0])
        downside_scenarios = len([r for r in final_returns if r < 0])
        total_scenarios = len(final_returns)
        
        risk_metrics = {
            "var_95": var_95,
            "var_99": var_99,
            "probability_profit": upside_scenarios / total_scenarios if total_scenarios else 0,
            "probability_loss": downside_scenarios / total_scenarios if total_scenarios else 0,
            "expected_shortfall_95": statistics.mean(sorted_returns[:int(0.05 * len(sorted_returns))]) if sorted_returns else 0
        }
        
        return {
            "simulations": simulation_results[:10],
            "statistics": statistics_dict,
            "risk_metrics": risk_metrics,
            "total_simulations": simulations
        }
    
    @staticmethod
    def regime_switching_analysis(prices):
        """Advanced Regime Switching Analysis (Safe Calculations)"""
        if len(prices) < 100:
            return {"regimes": [], "current_regime": 0, "transition_probabilities": []}
        
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] != 0:  # Avoid division by zero
                returns.append((prices[i] - prices[i-1]) / prices[i-1])
        
        if len(returns) < 20:
            return {"regimes": [], "current_regime": 0, "transition_probabilities": []}
        
        # Volatility-based regime identification
        window_size = 20
        volatilities = []
        
        for i in range(window_size, len(returns)):
            window_returns = returns[i-window_size:i]
            if len(window_returns) > 1:
                vol = statistics.stdev(window_returns)
            else:
                vol = 0
            volatilities.append(vol)
        
        if not volatilities:
            return {"regimes": [], "current_regime": 0, "transition_probabilities": []}
        
        vol_median = statistics.median(volatilities)
        vol_threshold = vol_median * 1.5 if vol_median else 0
        
        regimes = []
        for vol in volatilities:
            if vol > vol_threshold:
                regimes.append(1)  # High volatility regime
            else:
                regimes.append(0)  # Low volatility regime
        
        # Transition probabilities
        transitions = {"00": 0, "01": 0, "10": 0, "11": 0}
        
        for i in range(1, len(regimes)):
            prev_regime = regimes[i-1]
            curr_regime = regimes[i]
            key = f"{prev_regime}{curr_regime}"
            transitions[key] += 1
        
        total_transitions = sum(transitions.values())
        
        if total_transitions > 0:
            transition_probs = {k: v / total_transitions for k, v in transitions.items()}
        else:
            transition_probs = {"00": 0.5, "01": 0.25, "10": 0.25, "11": 0.5}
        
        current_regime = regimes[-1] if regimes else 0
        
        # Regime characteristics
        regime_stats = {}
        for regime_id in [0, 1]:
            regime_returns = [returns[i] for i in range(len(regimes)) if regimes[i] == regime_id]
            if regime_returns:
                regime_stats[regime_id] = {
                    "mean_return": statistics.mean(regime_returns),
                    "volatility": statistics.stdev(regime_returns) if len(regime_returns) > 1 else 0,
                    "duration": len(regime_returns) / len(regimes) if regimes else 0
                }
            else:
                regime_stats[regime_id] = {"mean_return": 0, "volatility": 0, "duration": 0}
        
        return {
            "regimes": regimes,
            "current_regime": current_regime,
            "transition_probabilities": transition_probs,
            "regime_statistics": regime_stats,
            "regime_interpretation": {
                0: "بازار پایدار (نوسان کم)",
                1: "بازار پرنوسان (ریسک بالا)"
            }
        }

class TechnicalAnalysisEngine:
    """Complete Technical Analysis Engine (Safe Calculations)"""
    
    @staticmethod
    def calculate_sma(prices, period):
        """Simple Moving Average"""
        if len(prices) < period:
            return 0
        return sum(prices[-period:]) / period
    
    @staticmethod
    def calculate_ema(prices, period):
        """Exponential Moving Average"""
        if len(prices) < period or period <= 0:
            return 0
            
        k = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price * k) + (ema * (1 - k))
        return ema
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        """Relative Strength Index (Safe Calculation)"""
        if len(prices) < period + 1:
            return 50
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100 if avg_gain != 0 else 50
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(prices):
        """MACD calculation (Safe)"""
        if len(prices) < 26:
            return 0, 0, 0
        
        ema12 = TechnicalAnalysisEngine.calculate_ema(prices, 12)
        ema26 = TechnicalAnalysisEngine.calculate_ema(prices, 26)
        macd_line = ema12 - ema26
        
        macd_values = []
        for i in range(26, len(prices)):
            temp_ema12 = TechnicalAnalysisEngine.calculate_ema(prices[:i+1], 12)
            temp_ema26 = TechnicalAnalysisEngine.calculate_ema(prices[:i+1], 26)
            macd_values.append(temp_ema12 - temp_ema26)
        
        if len(macd_values) >= 9:
            signal_line = TechnicalAnalysisEngine.calculate_ema(macd_values, 9)
        else:
            signal_line = 0
            
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(prices, period=20):
        """Bollinger Bands (Safe)"""
        if len(prices) < period:
            return 0, 0, 0
        
        sma = TechnicalAnalysisEngine.calculate_sma(prices, period)
        squared_diffs = [(price - sma) ** 2 for price in prices[-period:]]
        variance = sum(squared_diffs) / period
        std_dev = math.sqrt(variance) if variance > 0 else 0
        
        upper_band = sma + (2 * std_dev)
        lower_band = sma - (2 * std_dev)
        
        return sma, upper_band, lower_band
    
    @staticmethod
    def calculate_stochastic(prices, period=14):
        """Stochastic Oscillator (Safe)"""
        if len(prices) < period:
            return 50, 50
        
        recent_prices = prices[-period:]
        highest = max(recent_prices)
        lowest = min(recent_prices)
        current = prices[-1]
        
        if highest == lowest:
            return 50, 50
        
        k_percent = ((current - lowest) / (highest - lowest)) * 100
        
        k_values = []
        for i in range(period, len(prices)):
            temp_prices = prices[i-period:i]
            if not temp_prices:
                continue
                
            temp_high = max(temp_prices)
            temp_low = min(temp_prices)
            temp_current = prices[i]
            
            if temp_high != temp_low:
                temp_k = ((temp_current - temp_low) / (temp_high - temp_low)) * 100
                k_values.append(temp_k)
        
        if len(k_values) >= 3:
            d_percent = statistics.mean(k_values[-3:])
        elif k_values:
            d_percent = statistics.mean(k_values)
        else:
            d_percent = k_percent
        
        return k_percent, d_percent

# ==================== بخش ربات تلگرام ====================
async def fetch_crypto_data(coin_id: str, days: int = 365):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {'vs_currency': 'usd', 'days': days}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'prices': [point[1] for point in data['prices']],
                        'volumes': [point[1] for point in data['total_volumes']],
                        'market_caps': [point[1] for point in data['market_caps']]
                    }
                else:
                    raise Exception(f"خطای API: {response.status}")
    except Exception as e:
        raise Exception(f"خطا در دریافت داده: {str(e)}")

async def start(update: Update, context: CallbackContext):
    await update.message.reply_text(
        "سلام! به ربات تحلیل پیشرفته رمزارزها خوش آمدید.\n\n"
        "دستورات:\n"
        "/analyze <coin> - تحلیل کامل یک رمزارز (مثلاً: /analyze bitcoin)\n"
        "/help - راهنمایی"
    )

async def help_command(update: Update, context: CallbackContext):
    await update.message.reply_text(
        "📝 راهنمای ربات:\n\n"
        "1. برای تحلیل یک ارز از دستور زیر استفاده کنید:\n"
        "   /analyze bitcoin\n\n"
        "2. ارزهای پشتیبانی شده: هر ارزی که در CoinGecko وجود دارد\n"
        "3. مثالها:\n"
        "   /analyze ethereum\n"
        "   /analyze dogecoin\n\n"
        "⚠️ توجه: تحلیل ممکن است 10-20 ثانیه زمان ببرد"
    )

async def analyze_command(update: Update, context: CallbackContext):
    if not context.args:
        await update.message.reply_text("لطفاً نام ارز را وارد کنید. مثال: /analyze bitcoin")
        return
    
    coin_id = context.args[0].lower()
    user = update.effective_user
    await update.message.reply_text(
        f"سلام {user.first_name}!\n"
        f"در حال تحلیل {coin_id.upper()}... لطفاً 10-20 ثانیه منتظر بمانید ⏳"
    )
    
    try:
        # دریافت داده
        data = await fetch_crypto_data(coin_id)
        prices = data['prices']
        volumes = data['volumes']
        current_price = prices[-1] if prices else 0
        
        if not prices:
            await update.message.reply_text("ارز مورد نظر یافت نشد یا داده کافی ندارد.")
            return
        
        # ایجاد موتورها
        tech_engine = TechnicalAnalysisEngine()
        ai_engine = AdvancedAIEngine()
        quant_engine = AdvancedQuantEngine()
        
        # ===== تحلیل تکنیکال =====
        rsi = tech_engine.calculate_rsi(prices)
        macd_line, signal_line, _ = tech_engine.calculate_macd(prices)
        sma20 = tech_engine.calculate_sma(prices, 20)
        k_percent, d_percent = tech_engine.calculate_stochastic(prices)
        
        tech_report = (
            f"📊 [تحلیل تکنیکال {coin_id.upper()}]\n"
            f"• قیمت فعلی: ${current_price:.4f}\n"
            f"• RSI (14): {rsi:.1f} {'(فروش بیش‌حد 🔴)' if rsi > 70 else '(خرید بیش‌حد 🟢)' if rsi < 30 else '(نرمال ⚪️)'}\n"
            f"• MACD: {macd_line:.4f} | Signal: {signal_line:.4f}\n"
            f"• MA20: ${sma20:.4f}\n"
            f"• Stochastic: %K={k_percent:.1f}, %D={d_percent:.1f}\n\n"
        )
        
        # ===== تحلیل هوش مصنوعی =====
        ai_result = ai_engine.ensemble_prediction(prices, volumes)
        pattern_result = ai_engine.pattern_recognition_ai(prices)
        
        ai_report = (
            f"🤖 [تحلیل هوش مصنوعی {coin_id.upper()}]\n"
            f"• سیگنال: {'🟢 خرید' if ai_result['ensemble_signal']=='BUY' else '🔴 فروش' if ai_result['ensemble_signal']=='SELL' else '⚪️ نگهداری'}\n"
            f"• اعتماد: {ai_result['confidence']:.1f}%\n"
            f"• الگوها: {', '.join(pattern_result['patterns']) if pattern_result['patterns'] else 'هیچ الگوی مهمی یافت نشد'}\n"
        )
        
        # ===== تحلیل کمی =====
        quant_result = quant_engine.monte_carlo_simulation(prices)
        regime_result = quant_engine.regime_switching_analysis(prices)
        current_regime = regime_result['regime_interpretation'].get(regime_result.get('current_regime', 0), "نامشخص")
        
        quant_report = (
            f"🧮 [تحلیل کمی {coin_id.upper()}]\n"
            f"• بازده مورد انتظار (30 روز): {quant_result['statistics']['mean_return']*100:+.2f}%\n"
            f"• احتمال سود: {quant_result['risk_metrics']['probability_profit']*100:.1f}%\n"
            f"• رژیم بازار: {current_regime}\n"
            f"• VaR (95%): {quant_result['risk_metrics']['var_95']*100:.2f}%\n"
        )
        
        # ===== خلاصه‌گیری =====
        summary = (
            f"🌟 [خلاصه تحلیل {coin_id.upper()}]\n"
            f"➖➖➖➖➖➖➖➖➖\n"
            f"🔹 تکنیکال: {'خرید 🟢' if rsi < 30 else 'فروش 🔴' if rsi > 70 else 'خنثی ⚪️'}\n"
            f"🔹 هوش مصنوعی: {'خرید 🟢' if ai_result['ensemble_signal']=='BUY' else 'فروش 🔴' if ai_result['ensemble_signal']=='SELL' else 'خنثی ⚪️'}\n"
            f"🔹 کمی: {'مثبت 📈' if quant_result['statistics']['mean_return'] > 0 else 'منفی 📉'}\n\n"
            f"✅ توصیه نهایی: {'خرید 🟢' if ai_result['ensemble_signal']=='BUY' and rsi < 40 else 'فروش 🔴' if ai_result['ensemble_signal']=='SELL' and rsi > 60 else 'نگهداری ⚪️'}\n\n"
            f"🕒 زمان تحلیل: {datetime.now().strftime('%H:%M:%S')}"
        )
        
        # ارسال گزارش
        await update.message.reply_text(tech_report)
        await asyncio.sleep(1)
        await update.message.reply_text(ai_report)
        await asyncio.sleep(1)
        await update.message.reply_text(quant_report)
        await asyncio.sleep(1)
        await update.message.reply_text(summary)
        
    except Exception as e:
        await update.message.reply_text(f"⚠️ خطا در تحلیل: {str(e)}")

def main():
    print("Starting Crypto Analysis Bot...")
    app = Application.builder().token(TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("analyze", analyze_command))
    
    print("Bot is running...")
    app.run_polling()
    print("Bot stopped")

if __name__ == '__main__':
    main()