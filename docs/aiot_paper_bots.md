# AIOT paper bots

Seven Hummingbot V2 controllers paper-trade AIOT-USDT. They are long only. Each config is its own bot. The controller is `aiot_research` (`bots/controllers/directional_trading/aiot_research.py`) and the rules live in `aiot_engine.py`, which the unit tests drive without Hummingbot installed.

Orders go to the paper connector `mexc_paper_trade`. Candles, including the higher-timeframe filter, come from the public `mexc` spot feed, and only from candles that have closed. A signal is read on that close. The paper bot sends the market order immediately after. No live MEXC API key is required, and this repo does not add one.

## The seven configs

| Config | Strategy | Notional | What it does |
| --- | --- | --- | --- |
| `aiot_f1` | `f1` | 15000 USDT | 1h EMA20/50 trend-hold on 15m closes |
| `aiot_f3` | `f3` | 15000 USDT | 15m RSI 28 single |
| `aiot_rsi32` | `rsi32` | 15000 USDT | 15m RSI 32 single |
| `aiot_ladder` | `ladder` | 15000 USDT | 15m RSI ladder, 5000 per rung |
| `aiot_fin_a` | `fin_a` | 12000 + 3000 USDT | F3 sleeve (RSI exit 75) and F1-style trend sleeve together |
| `aiot_fin_b` | `fin_b` | 15000 USDT | F1 entry with an ATR stop and a 67% partial |
| `aiot_s2326` | `s2326` | 5000 USDT | 5m z-score sleeve |

Configs live in `bots/conf/controllers/`. `stop_loss`, `take_profit`, `time_limit`, and `trailing_stop` are left empty on purpose. The engine owns those exits. See "Why this is not the stock triple barrier" below.

### aiot_f1

Regime is EMA20 > EMA50 on the last closed 1h bar. Equality is off. On any closed 15m bar with the regime on and the sleeve flat, buy 15000 USDT. Exit when the last closed 1h bar has EMA20 <= EMA50, when price trades 15% below the fill, or when a closed 15m close is 30% below the highest 15m close since the fill. There is no time stop.

After a stop or a trail fills, the regime must be observed off and then on again before a new entry. A cross-down exit is not a stop or a trail, so the next time the regime is on is a valid entry. The cross itself is the off observation.

### aiot_f3

On a 15m close with RSI14 <= 28 and EMA50 > EMA200, buy 15000 USDT. Exit at RSI14 >= 70, at a stop 10% below the fill, or 24 hours after the fill, whichever comes first.

### aiot_rsi32

Same as F3, with the entry at RSI14 <= 32.

### aiot_ladder

Rung 1 buys 5000 USDT at RSI14 <= 32, and only while EMA50 > EMA200. Rungs 2 and 3 add 5000 at RSI <= 26 and 5000 at RSI <= 20. The adds do not re-check the trend. The cap is 15000. If one close has RSI <= 20 while the book is flat and the trend is on, all three rungs fire.

The whole ladder exits at RSI >= 70, at a stop 10% below the first fill, or 24 hours after the first fill. After a full exit the rungs reset. A later close can start again at rung 1, including on the same bar as an intra-bar stop.

### aiot_fin_a

One controller, two sleeves, one paper balance.

The 12000 sleeve is F3 with the RSI exit at 75: entry RSI14 <= 28 with EMA50 > EMA200, exit at RSI14 >= 75, a 10% stop, or 24 hours.

The 3000 sleeve uses F1's entry. It exits on the 1h EMA20 <= EMA50 cross, at a hard stop of the fill minus 5 times the 1h ATR14 of the signal bar, on the 30% trailing stop, and it sells 50% when price trades at double the fill. F1's re-arm rule applies to this sleeve. The ATR stop is frozen from the signal bar.

### aiot_fin_b

F1's entry, the cross-down exit, the 30% trail, and the re-arm rule. The stop is the fill minus 5 times the signal bar's 1h ATR14. It sells 67% of the position when price trades at 2.5 times the fill. Notional 15000 USDT. The remainder stays under the stop, the trail, and the cross.

### aiot_s2326

5m bars. Enter when z = (close - SMA48) / population stdev of those 48 closes is <= -2.5, and the last closed 15m bar has EMA50 > EMA200. Take-profit is a resting limit at the fill times 1.03. Stop is 10% below the fill. Time stop is 288 closed 5m bars (24 hours of 5m bars when the feed has no gaps). There is no RSI exit. Notional is 5000 USDT.

## Indicators

RSI14 is Wilder's: the first average is the SMA of the first 14 changes, then the Wilder smooth. ATR14 is Wilder's on 1h true range, with the same SMA seed. EMAs are the standard recursive form with `adjust=False`: the first seed is the first close, alpha is 2 / (span + 1), and a value is withheld until `span` closes exist. The value at that bar is the recursive EMA, not an SMA seed. The z-score uses the population standard deviation (divide by N). A window with zero deviation is not a signal.

A higher-timeframe value is the last HTF candle whose close is at or before the signal candle's close. A 15m bar that closes on the hour may use the 1h bar that closes then. A 15m bar inside a forming hour uses the previous closed hour. The same rule aligns 5m bars to 15m. A forming HTF candle is ignored.

## Paper account

`bots/credentials/master_account/conf_client.yml` lists `mexc` in `paper_trade.paper_trade_exchanges` and sets the paper USDT balance to 200000. Deploy copies that profile into the instance, so a bot created after this change receives it. 200000 covers every sleeve at once (six times 15000 plus the 5000 five-minute sleeve, 95000 in total) and leaves quote for simulated fees. Each bot gets the full profile balance. Other paper exchanges and the BTC balance are unchanged.

`mexc_paper_trade` builds a non-trading MEXC connector for the public order book and wraps it in Hummingbot's paper exchange. No `bots/credentials/.../connectors/mexc.yml` key file is created.

If that public book fails to start, leave the orders on `mexc_paper_trade`. AIOT-USDT is not the book on the other paper exchanges in this profile, so pointing the orders elsewhere would paper-trade a different market. Candles stay on the connector name `mexc` either way.

## Image

Condor and `POST /bot-orchestration/deploy-v2-controllers` default the bot image to `hummingbot/hummingbot:latest`. That default is not this fork. Set the image to one built from [ayj777/hummingbot](https://github.com/ayj777/hummingbot) at commit `6e4f17ebcda5fe86c2358131bd488c9e4e756c39` (the pin in `environment.yml`). This repo does not publish a Docker tag for that commit.

The controller Python is bind-mounted from this repo's `bots/controllers` onto `/home/hummingbot/controllers`, so it does not have to be baked into the image. The image still has to contain the MEXC connector, which that commit does.

## Deploy from Condor

1. Run this API with the `master_account` profile above. Restart is not required for `conf_client.yml` if you deploy after the edit: the file is copied at deploy time.
2. Open the Bots page and choose Deploy Bot.
3. Select exactly one of `aiot_f1`, `aiot_f3`, `aiot_rsi32`, `aiot_ladder`, `aiot_fin_a`, `aiot_fin_b`, `aiot_s2326`. One config per bot. FIN-A is already both sleeves.
4. In the advanced settings, set Account to `master_account`. Set Image to the fork image from the commit above.
5. Deploy.

The API renames the instance to `{instance_name}-{YYYYMMDD-HHMMSS}`. Use that name when you read status and history.

## Deploy from the API

`POST /bot-orchestration/deploy-v2-controllers` with the API's basic auth:

```json
{
  "instance_name": "aiot-f1",
  "credentials_profile": "master_account",
  "controllers_config": ["aiot_f1"],
  "image": "<image built from ayj777/hummingbot@6e4f17ebcda5fe86c2358131bd488c9e4e756c39>"
}
```

Swap the config name for each of the other six bots. Do not send live keys. The response field `unique_instance_name` is the running bot.

## Reading results

- The Condor Bots page shows pnl, volume, and controller status.
- `GET /bot-orchestration/status`
- `GET /bot-orchestration/{unique_instance_name}/status`
- `GET /bot-orchestration/{unique_instance_name}/history`
- `GET /bot-orchestration/controller-performance-latest?bot_name={unique_instance_name}`
- `GET /bot-orchestration/controller-performance-history?bot_name={unique_instance_name}`
- `GET /performance/latest` and `GET /performance/history`

The controller performance report includes `get_custom_info()`: strategy, connector, candles connector, pair, and each sleeve (`phase`, `base`, `first_fill_price`, `stop_price`, `highest_close`, `blocked`, `rungs`). Instance logs are in the bot's `logs` directory.

Sleeve state is written to `data/aiot_research_{config_id}.json` inside the container, which is `bots/instances/{unique_instance_name}/data/` on the host. Deleting that file while a paper position is open makes the next signal look flat and can open a second position. The controller does not rebuild the sleeve from the paper balance.

## Why this is not the stock triple barrier

The stock position-executor barrier would not match these rules:

- Its stop is net pnl, fees included, so a 10% or 15% price stop would fire early. MEXC's default fee on this fork is 0.05% per side, and the buy fee is deducted from the returned base.
- Its trail follows live pnl, not the highest closed signal bar.
- Its take-profit closes the whole position, so a 50% or 67% partial, a ladder, and a stop measured from the first fill cannot be expressed.
- The re-arm rule is controller state.

Entries, stop exits, cross exits, RSI exits, and time exits are market `OrderExecutor`s. Partials and the S2326 take-profit are resting limit sells. The engine is the only decision maker. `leverage` is 1 and `position_mode` is `HEDGE`. Buys use `PositionAction.OPEN`. Sells use `PositionAction.CLOSE`, so a hedge-mode sell closes the long. `size_mode` is `max`: the whole position is capped at `total_amount_quote`, at the available quote times leverage times 0.95, and at 98 percent of one exchange order. Relative sleeve weights stay the fractions of that cap. The exchange stop covers that position until the live long is flat.

## Where the paper bot differs from the tests

`pytest test/test_aiot_engine.py` runs the engine on synthetic candles. It does not need Hummingbot. These are the places the paper bot cannot be identical to that fill model:

1. Tests fill a market order at the next bar's open. The paper bot sends the market order as soon as the bar has closed, so the fill is the paper book at that moment. Stops and targets are measured from the actual fill. Size is base = notional / signal close, so quote spent is base times the fill and can differ from the round notional.
2. A hard stop in the bot is a market sell when the mid trades at or below the stop. On a connector that can place one, the same hard stop is also a reduce-only exchange stop, resized after a partial and cancelled when the position is flat. The trail stays in the bot and does not move that exchange stop. The tests fill at the stop, or at the open when the bar opens through it. The paper fill is about the mid at the trigger.
3. When one candle trades both a resting limit and the stop, the tests use a fixed path. A bar that closes at or above its open is treated as low then high, so the stop exits the whole position and the limit does not fill. A bar that closes below its open is treated as high then low, so the limit fills first and the stop sells the remainder. A gap open through the stop fills the stop at the open. A gap open through the limit fills the limit at the open. The paper book sequences those trades in the order they print.
4. The trail fires only when a closed signal-bar close is at or below (1 - 0.30) times the highest close since the fill. The peak starts at the first close after the fill. A wick does not trail out, and the live mid check does not either. The live mid check is the hard stop only.
5. Amounts are quantized to the connector's increment, so 67% and a 5000 rung can differ by one step. The engine uses exact floats.
6. A restart reloads the state file and skips executor ids it has already applied. An entry that was still in flight reloads as flat, and the controller holds the sleeve pending while that buy executor is still active so the next close does not send a second buy. An exit that was still in flight reloads as open and is held pending while the sell is active. If the order is gone, the next closed bar can signal again. A resting limit that is still active is not placed a second time.
7. The 15m time stop is 24 hours of clock time from the fill, then a market sell. S2326 is 288 closed 5m bars. A gap in the 5m feed makes 288 bars longer than 24 hours.
8. On the first tick the controller acts on the latest closed signal bar only. It does not replay older bars.
9. If several close-based exits are true on the same bar, the order is trail, then the regime cross, then RSI, then the time stop. A trail or a stop rearms. A cross, an RSI exit, and a time stop do not.
10. Reported paper pnl is after simulated fees. The stop trigger itself is on price.

## Shared live bots

`aiot_shared` (`bots/controllers/directional_trading/aiot_shared.py`) is a second controller for the same engine. It is for two or more bots on one MEXC perpetual account and the same AIOT-USDT hedge long. It also runs alone, after the bots move to their own sub-accounts. The paper controller `aiot_research` is unchanged and still assumes it owns the whole long.

Each shared bot tracks only its own base, from fills on its own client order ids. That share is stored in `data/aiot_shared_{config_id}.json` and checked against the bot's order history on startup. If the history is missing or the sizes do not match, the bot halts entries and exits and says why. It does not adopt an existing long, and it does not cancel or replace another bot's orders or stop plans.

Every exit is a reduce-only close of this bot's own base, and only when that base is not larger than the exchange long. If the exchange long is smaller than the share, the bot halts. It does not close the residual, because that residual may belong to the other bot. A market sell is not sent for a hard stop while this bot's exchange stop is still open, and any other exit cancels that stop before the sell is sent. The exchange stop is its own partial plan (`volType` 1), sized to its own base at its own stop price. It is left in place while the size and price are unchanged. A partial exit cancels and replaces that plan. A full exit cancels only that plan.

MEXC does allow several partial stop plans on one long. Two limits still apply. The new volume plus the volume of stop plans already on the position must be less than the closable quantity, so two plans that add up to the whole long can be rejected. A new plan at the same price does not create a second order: the exchange returns the existing id and updates that plan. This controller will not adopt or cancel that id, and it keeps new entries off until it has a stop of its own.

`size_mode: max` sizes each order to the minimum of that bot's `total_amount_quote`, the available quote times leverage times 0.95, and 98 percent of one MEXC order. Available quote already excludes margin the other bot is using. Entries still wait until 1x isolated is confirmed, and the account must stay HEDGE.

| Config | Strategy | Order size | What it does |
| --- | --- | --- | --- |
| `aiot_fin_b_shared` | `fin_b` | 2200 USDT | The same fin_b rules as the paper bot, on this bot's share only |
| `aiot_f3l20_shared` | `f3l20` | 1650 USDT per leg | F3 entry, one add at RSI14 <= 20, up to 3300 USDT |

F3-L20 buys one `total_amount_quote` on a 15m close with RSI14 <= 28 and EMA50 > EMA200. While that leg is open, one later 15m close with RSI14 <= 20 buys the same amount again and does not re-check the trend. The engine sells the whole share at the next open after RSI14 >= 70, at a stop 10 percent under the average fill (the exchange stop is replaced after the second leg), or 24 hours after the first fill. The live bot sends that market order when the 15m bar closes, the same as the paper bot.

The two configs together can hold 5500 USDT (2200 plus two legs of 1650). Deploy them like the paper bots, with connector `mexc_perpetual`, leverage 1, position mode HEDGE, and one config per bot.

The live fin_b bot on `aiot_research` sizes its stop and its exits to the whole exchange long. Stop that bot and flatten its long before starting `aiot_fin_b_shared`, and do that before starting F3-L20. `aiot_shared` will not take over the open long. Leaving the old controller running will close or stop the other bot's share.
