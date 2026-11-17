import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { DatabaseSync } from "node:sqlite";
import { mkdirSync, readFileSync, readdirSync } from "node:fs";

//#region scripts/radar-local-db.mjs
function openResearchDb(filename) {
	mkdirSync(dirname(filename), { recursive: true });
	const sqlite = new DatabaseSync(filename);
	sqlite.exec("PRAGMA journal_mode=WAL; CREATE TABLE IF NOT EXISTS local_migrations(name TEXT PRIMARY KEY); CREATE TABLE IF NOT EXISTS local_workflow_state(name TEXT PRIMARY KEY,value TEXT);");
	const migrations = new URL("../cloudflare/radar/migrations/", import.meta.url);
	for (const name of readdirSync(migrations).filter((name) => name.endsWith(".sql")).sort()) {
		if (sqlite.prepare("SELECT name FROM local_migrations WHERE name=?").get(name)) continue;
		sqlite.exec("BEGIN");
		try {
			sqlite.exec(readFileSync(new URL(name, migrations), "utf8"));
			sqlite.prepare("INSERT INTO local_migrations VALUES(?)").run(name);
			sqlite.exec("COMMIT");
		} catch (error) {
			sqlite.exec("ROLLBACK");
			throw error;
		}
	}
	return {
		sqlite,
		prepare(sql) {
			const statement = sqlite.prepare(sql);
			let args = [];
			return {
				bind(...values) {
					args = values;
					return this;
				},
				async first() {
					return statement.get(...args) || null;
				},
				async all() {
					return { results: statement.all(...args) };
				},
				async run() {
					return { meta: { changes: Number(statement.run(...args).changes) } };
				}
			};
		},
		async batch(statements) {
			sqlite.exec("BEGIN");
			try {
				const results = [];
				for (const statement of statements) results.push(await statement.run());
				sqlite.exec("COMMIT");
				return results;
			} catch (error) {
				sqlite.exec("ROLLBACK");
				throw error;
			}
		},
		close() {
			sqlite.exec("PRAGMA wal_checkpoint(TRUNCATE)");
			sqlite.close();
		}
	};
}

//#endregion
//#region src/radar/config.js
const freeze = (value) => {
	Object.values(value).forEach((item) => {
		if (item && typeof item === "object") freeze(item);
	});
	return Object.freeze(value);
};
const strategyConfig = freeze({
	version: "radar-paper-v1",
	mode: "RESEARCH",
	holdingHorizons: [
		5,
		10,
		20
	],
	minimumBars: 141,
	minimumOosSamples: 50,
	weights: {
		relativeStrength: 20,
		trend: 15,
		setup: 20,
		momentumPersistence: 10,
		volume: 10,
		institutional: 10,
		fundamental: 7.5,
		newsCatalyst: 7.5
	},
	liquidity: {
		minimumPrice: 5,
		maximumPrice: 1e4,
		minimumAvg20Volume: 2e5,
		minimumAvg20Turnover: 2e7,
		maximumSpreadPct: 1
	},
	thresholds: {
		Bull: 70,
		Neutral: 76,
		"Risk-Off": 84
	},
	watchDiscount: 12,
	maximumPicks: 5,
	maximumWatch: 20,
	pullback: {
		minimumDays: 3,
		maximumDays: 10,
		minimumDrawdown: 2,
		maximumDrawdown: 12,
		minimumRsi: 35,
		maximumRsi: 55,
		maximumMa20Distance: 5
	},
	breakout: {
		windows: [20, 55],
		minimumVolumeRatio: 1.3,
		maximumMa20Distance: 10
	},
	risk: {
		maximumGapPct: 5,
		maximumFiveDayReturn: 20,
		maximumTenDayReturn: 30,
		maximumRsi: 80,
		maximumMa20Distance: 15,
		maximumAtrPct: 9,
		maximumUpperWickRatio: .6
	},
	exits: {
		atrMultiple: 2,
		trailingAtrMultiple: 2,
		profitAtrMultiple: 3,
		maximumHoldingDays: 20
	},
	deterioration: {
		minimumSamples: 30,
		minimumProfitFactor: 1,
		multiplier: .65
	},
	freshness: {
		maximumMarketDays: 4,
		maximumNewsHours: 72
	},
	schedule: {
		timezone: "Asia/Taipei",
		collection: "04:10–04:40",
		collectionTimes: ["04:10", "04:40"],
		initial: "04:55",
		advanceUntil: "05:55",
		publish: "06:00",
		refreshFrom: "05:55",
		refreshUntil: "06:10"
	}
});
const TradingCostConfig = freeze({
	version: "tw-common-paper-v1",
	notional: 1e5,
	commissionRate: .001425,
	minimumCommission: 20,
	sellTaxRate: .003,
	slippageBps: 10,
	maximumVolumeParticipation: .01
});
function validateConfig(config) {
	const weights = Object.values(config.weights || {});
	if (weights.length !== 8 || weights.some((value) => !Number.isFinite(value) || value < 0) || Math.abs(weights.reduce((sum, value) => sum + value, 0) - 100) > 1e-6) throw Error("INVALID_WEIGHTS");
	if (!(config.thresholds.Bull <= config.thresholds.Neutral && config.thresholds.Neutral <= config.thresholds["Risk-Off"]) || config.minimumOosSamples < 50) throw Error("INVALID_STRATEGY_CONFIG");
	return config;
}

//#endregion
//#region src/radar/math.js
const mean = (values) => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
const clamp = (value, minimum = 0, maximum = 100) => Math.max(minimum, Math.min(maximum, value));
const finite = (value) => typeof value === "number" && Number.isFinite(value);
const round = (value) => finite(value) ? Math.round(value * 100) / 100 : null;
const returnPct = (bars, days) => bars.length > days ? (bars.at(-1).close / bars.at(-1 - days).close - 1) * 100 : null;
const ma = (bars, days, end = bars.length) => end >= days ? mean(bars.slice(end - days, end).map((bar) => bar.close)) : null;
function rsi(bars, period = 14) {
	if (bars.length <= period) return null;
	const changes = bars.slice(1).map((bar, index) => bar.close - bars[index].close);
	let gain = mean(changes.slice(0, period).map((value) => Math.max(0, value))), loss = mean(changes.slice(0, period).map((value) => Math.max(0, -value)));
	for (const change of changes.slice(period)) {
		gain = (gain * (period - 1) + Math.max(change, 0)) / period;
		loss = (loss * (period - 1) + Math.max(-change, 0)) / period;
	}
	return gain === 0 && loss === 0 ? 50 : loss === 0 ? 100 : 100 - 100 / (1 + gain / loss);
}
function atr(bars, period = 14) {
	if (bars.length <= period) return null;
	const ranges = bars.slice(1).map((bar, index) => Math.max(bar.high - bar.low, Math.abs(bar.high - bars[index].close), Math.abs(bar.low - bars[index].close)));
	let value = mean(ranges.slice(0, period));
	for (const range of ranges.slice(period)) value = (value * (period - 1) + range) / period;
	return value;
}
function adx(bars, period = 14) {
	if (bars.length < period * 2 + 1) return null;
	const tr = [], plus = [], minus = [];
	for (let index = 1; index < bars.length; index++) {
		const bar = bars[index], previous = bars[index - 1], up = bar.high - previous.high, down = previous.low - bar.low;
		tr.push(Math.max(bar.high - bar.low, Math.abs(bar.high - previous.close), Math.abs(bar.low - previous.close)));
		plus.push(up > down && up > 0 ? up : 0);
		minus.push(down > up && down > 0 ? down : 0);
	}
	let smoothTr = mean(tr.slice(0, period)), smoothPlus = mean(plus.slice(0, period)), smoothMinus = mean(minus.slice(0, period));
	const dx = [];
	for (let index = period - 1; index < tr.length; index++) {
		if (index >= period) {
			smoothTr = (smoothTr * (period - 1) + tr[index]) / period;
			smoothPlus = (smoothPlus * (period - 1) + plus[index]) / period;
			smoothMinus = (smoothMinus * (period - 1) + minus[index]) / period;
		}
		const positive = smoothTr ? 100 * smoothPlus / smoothTr : 0, negative = smoothTr ? 100 * smoothMinus / smoothTr : 0;
		dx.push(positive + negative ? 100 * Math.abs(positive - negative) / (positive + negative) : 0);
	}
	if (dx.length < period) return null;
	let value = mean(dx.slice(0, period));
	for (const point of dx.slice(period)) value = (value * (period - 1) + point) / period;
	return value;
}
function correlation(left, right) {
	if (left.length !== right.length || left.length < 3) return null;
	const avgLeft = mean(left), avgRight = mean(right);
	const covariance = left.reduce((sum, value, index) => sum + (value - avgLeft) * (right[index] - avgRight), 0), varianceLeft = left.reduce((sum, value) => sum + (value - avgLeft) ** 2, 0), varianceRight = right.reduce((sum, value) => sum + (value - avgRight) ** 2, 0);
	return varianceLeft && varianceRight ? covariance / Math.sqrt(varianceLeft * varianceRight) : null;
}
function visibleRows(rows, asOf) {
	const cutoff = Date.parse(asOf);
	if (!finite(cutoff)) throw Error("INVALID_AS_OF");
	return (rows || []).filter((row) => finite(Date.parse(row.availableAt)) && Date.parse(row.availableAt) <= cutoff && /^\d{4}-\d{2}-\d{2}$/.test(row.date || "") && Date.parse(row.date) <= cutoff).sort((left, right) => left.date.localeCompare(right.date) || Date.parse(right.availableAt) - Date.parse(left.availableAt)).filter((row, index, all) => index === 0 || row.date !== all[index - 1].date);
}
function validBar(bar) {
	return [
		"open",
		"high",
		"low",
		"close",
		"volume",
		"turnover"
	].every((field) => finite(bar[field])) && bar.volume >= 0 && bar.turnover >= 0 && bar.low > 0 && bar.high >= Math.max(bar.open, bar.close) && bar.low <= Math.min(bar.open, bar.close);
}

//#endregion
//#region src/radar/engine.js
function technicalSnapshot(rows, asOf) {
	const bars = visibleRows(rows, asOf).filter(validBar), last = bars.at(-1), previous = bars.at(-2);
	if (!last) return null;
	const returns = Object.fromEntries([
		5,
		10,
		20,
		60,
		120
	].map((days) => [days, returnPct(bars, days)])), ma20 = ma(bars, 20), ma60 = ma(bars, 60), ma120 = ma(bars, 120), volume20 = mean(bars.slice(-20).map((bar) => bar.volume)), previousVolume20 = mean(bars.slice(-21, -1).map((bar) => bar.volume));
	const daily = bars.slice(-21).slice(1).map((bar, index) => bar.close / bars.slice(-21)[index].close - 1), gains = daily.filter((value) => value > 0), losses = daily.filter((value) => value < 0);
	const highs = bars.slice(-11).map((bar) => bar.high), peak = Math.max(...highs), peakIndex = highs.lastIndexOf(peak), pullbackDays = highs.length - 1 - peakIndex;
	const pullbackBars = pullbackDays ? bars.slice(-pullbackDays) : [], advanceBars = pullbackDays ? bars.slice(-pullbackDays - 10, -pullbackDays) : [];
	const advanceDays = advanceBars.filter((bar, index) => index > 0 && bar.close > advanceBars[index - 1].close);
	return {
		bars,
		last,
		returns,
		ma20,
		ma60,
		ma120,
		rsi14: rsi(bars),
		atr14: atr(bars),
		adx14: adx(bars),
		ma20Slope: ma20 && ma(bars, 20, bars.length - 5) ? (ma20 / ma(bars, 20, bars.length - 5) - 1) * 100 : null,
		ma60Slope: ma60 && ma(bars, 60, bars.length - 5) ? (ma60 / ma(bars, 60, bars.length - 5) - 1) * 100 : null,
		avg20Volume: volume20,
		avg20Turnover: mean(bars.slice(-20).map((bar) => bar.turnover)),
		ma20Distance: ma20 ? (last.close / ma20 - 1) * 100 : null,
		ma60Distance: ma60 ? (last.close / ma60 - 1) * 100 : null,
		volumeRatio5: bars.length >= 6 && mean(bars.slice(-6, -1).map((bar) => bar.volume)) ? last.volume / mean(bars.slice(-6, -1).map((bar) => bar.volume)) : null,
		volumeRatio20: previousVolume20 ? last.volume / previousVolume20 : null,
		positiveCloseRatio: daily.length ? gains.length / daily.length : null,
		averageUp: mean(gains),
		averageDown: mean(losses),
		priceVolumeCorrelation: correlation(daily, bars.slice(-20).map((bar) => bar.volume)),
		drawdown: peak ? (peak - last.close) / peak * 100 : null,
		pullbackDays,
		pullbackVolume: mean(pullbackBars.map((bar) => bar.volume)),
		advanceVolume: mean(advanceDays.map((bar) => bar.volume)),
		previousHigh20: bars.length >= 21 ? Math.max(...bars.slice(-21, -1).map((bar) => bar.high)) : null,
		previousHigh55: bars.length >= 56 ? Math.max(...bars.slice(-56, -1).map((bar) => bar.high)) : null,
		gapPct: previous ? (last.open / previous.close - 1) * 100 : null,
		upperWickRatio: last.high > last.low ? (last.high - Math.max(last.open, last.close)) / (last.high - last.low) : 0,
		atrPct: last.close ? atr(bars) / last.close * 100 : null
	};
}
function eligibleSecurity(stock, snapshot, asOf, benchmarkDate, config = strategyConfig) {
	const reasons = [], warnings = [], rules = config.liquidity;
	if (stock.securityType !== "COMMON" || !["TWSE", "TPEx"].includes(stock.market)) reasons.push("非上市櫃普通股");
	if (!finite(Date.parse(stock.availableAt)) || Date.parse(stock.availableAt) > Date.parse(asOf) || stock.listedAt > asOf.slice(0, 10) || stock.delistedAt && stock.delistedAt <= asOf.slice(0, 10)) reasons.push("當時股票池資料不足");
	if (!snapshot || snapshot.bars.length < config.minimumBars) reasons.push("歷史日線不足");
	if (snapshot) {
		if (snapshot.last.close < rules.minimumPrice || snapshot.last.close > rules.maximumPrice) reasons.push("價格不符篩選範圍");
		if (snapshot.avg20Volume < rules.minimumAvg20Volume) reasons.push("20日均量不足");
		if (snapshot.avg20Turnover < rules.minimumAvg20Turnover) reasons.push("20日均成交金額不足");
		if (snapshot.last.volume <= 0 || snapshot.last.turnover <= 0 || snapshot.last.date !== benchmarkDate) reasons.push("最新交易資料不足或停牌");
		if (Date.parse(asOf) - Date.parse(snapshot.last.date) > config.freshness.maximumMarketDays * 864e5) reasons.push("行情過期");
		const recent = snapshot.bars.slice(-20);
		if (recent.some((bar, index) => index > 0 && Date.parse(bar.date) - Date.parse(recent[index - 1].date) > 12096e5)) reasons.push("近期長時間無交易");
	}
	const risk = visibleRows(stock.risk || [], asOf).at(-1);
	if (risk?.disposition) reasons.push("處置期間");
	if (finite(risk?.spreadPct) && risk.spreadPct > rules.maximumSpreadPct) reasons.push("買賣價差過大");
	if (!risk?.dispositionVerified) warnings.push("處置資訊未完整核實");
	if (!finite(risk?.spreadPct)) warnings.push("買賣價差未取得");
	return {
		eligible: reasons.length === 0,
		reasons,
		warnings
	};
}
function marketRegime(benchmark, asOf, context = {}, config = strategyConfig) {
	const snapshot = technicalSnapshot(benchmark || [], asOf), breadth = finite(context.breadth) ? context.breadth : null;
	if (!snapshot || snapshot.bars.length < 60) return {
		state: "Neutral",
		label: "資料不足",
		threshold: config.thresholds.Neutral,
		verified: false,
		date: snapshot?.last.date || null,
		breadth
	};
	const riskOff = snapshot.last.close < snapshot.ma60 || snapshot.last.close < snapshot.ma20 && snapshot.ma20Slope < 0 || finite(breadth) && breadth < .3;
	const bull = !riskOff && snapshot.last.close > snapshot.ma20 && snapshot.ma20 > snapshot.ma60 && snapshot.ma20Slope > 0 && finite(breadth) && breadth >= .5;
	const state = riskOff ? "Risk-Off" : bull ? "Bull" : "Neutral";
	return {
		state,
		label: {
			Bull: "偏多",
			Neutral: "中性",
			"Risk-Off": "防守"
		}[state],
		threshold: config.thresholds[state],
		verified: true,
		date: snapshot.last.date,
		breadth,
		close: snapshot.last.close,
		ma20: round(snapshot.ma20),
		ma60: round(snapshot.ma60),
		ma20Slope: round(snapshot.ma20Slope),
		volumeRatio20: round(snapshot.volumeRatio20)
	};
}
function institutionalSnapshot(rows, asOf, dates, averageVolume) {
	const visible = visibleRows(rows, asOf), byDate = new Map(visible.map((row) => [row.date, row]));
	const aggregate = (days) => {
		const selected = dates.slice(-days).map((date) => byDate.get(date));
		return Object.fromEntries([
			"foreign",
			"trust",
			"dealer"
		].map((field) => [field, selected.length === days && selected.every((row) => finite(row?.[field])) ? selected.reduce((sum, row) => sum + row[field], 0) : null]));
	};
	const sums = Object.fromEntries([
		3,
		5,
		10
	].map((days) => [days, aggregate(days)])), five = sums[5];
	let trustStreak = 0;
	for (const date of dates.slice().reverse()) {
		const point = byDate.get(date);
		if (!finite(point?.trust) || point.trust <= 0) break;
		trustStreak++;
	}
	const complete = Object.values(five).every(finite), normalized = complete && averageVolume ? Object.values(five).reduce((sum, value) => sum + value, 0) / (averageVolume * 5) : null;
	const score = complete ? clamp(40 + normalized * 100 + Math.min(trustStreak, 5) * 5 + (five.foreign > 0 && five.trust > 0 ? 15 : 0)) : null;
	return {
		sums,
		trustStreak,
		normalized5D: round(normalized),
		score: round(score),
		complete,
		asOf: visible.at(-1)?.date || null
	};
}
function fundamentalSnapshot(rows, asOf) {
	const monthly = /* @__PURE__ */ new Map();
	for (const row of visibleRows(rows, asOf).sort((left, right) => Date.parse(left.availableAt) - Date.parse(right.availableAt))) {
		let month = String(row.reportMonth || "").replace("-", "");
		if (/^\d{5}$/.test(month)) month = String(Number(month.slice(0, 3)) + 1911) + month.slice(3);
		if (/^\d{6}$/.test(month)) monthly.set(month, row);
	}
	const visible = [...monthly].sort(([left], [right]) => left.localeCompare(right)).map(([, row]) => row), last = visible.at(-1);
	if (!last || !finite(last.yoy) || Date.parse(asOf) - Date.parse(last.date) > 5616e6) return {
		score: null,
		complete: false
	};
	const keys = [...monthly.keys()].sort(), monthIndex = (key) => Number(key.slice(0, 4)) * 12 + Number(key.slice(4, 6)), consecutive = keys.length >= 3 && monthIndex(keys.at(-1)) - monthIndex(keys.at(-2)) === 1 && monthIndex(keys.at(-2)) - monthIndex(keys.at(-3)) === 1;
	const recent = visible.slice(-3), acceleration = visible.length > 1 && monthIndex(keys.at(-1)) - monthIndex(keys.at(-2)) === 1 && finite(visible.at(-2).yoy) ? last.yoy - visible.at(-2).yoy : null;
	return {
		score: clamp((last.yoy > 0 ? 35 : 0) + (last.yoy > 20 ? 15 : 0) + (last.mom > 0 ? 15 : 0) + (finite(acceleration) && acceleration > 0 ? 20 : 0) + (consecutive && recent.every((row) => row.yoy > 0) ? 15 : 0)),
		yoy: last.yoy,
		mom: last.mom,
		acceleration: round(acceleration),
		threeMonthTrend: consecutive ? recent.map((row) => row.yoy) : null,
		date: last.date,
		reportMonth: last.reportMonth,
		complete: consecutive
	};
}
function newsSnapshot(news, stock, asOf) {
	const events = (news || []).filter((event) => finite(Date.parse(event.availableAt)) && Date.parse(event.availableAt) <= Date.parse(asOf) && finite(Date.parse(event.publishedAt)) && Date.parse(event.publishedAt) <= Date.parse(asOf) && Date.parse(asOf) - Date.parse(event.publishedAt) <= 2592e5).filter((event) => (event.companies || []).some((company) => company.market === stock.market && company.symbol === stock.symbol)).map((event) => {
		const company = event.companies.find((item) => item.market === stock.market && item.symbol === stock.symbol);
		return {
			...event,
			direct: company.direct === true,
			direction: company.direction || event.direction || "uncertain"
		};
	});
	return {
		score: events.length ? Math.max(...events.map((event) => clamp((event.direct ? 35 : 10) + (/^(earnings|order|orders|partnership|capex|product|pricing|supply_chain|policy|mna)$/.test(event.eventType) ? 15 : 5) + (event.importance === "HIGH" ? 20 : 10) + (event.direction === "beneficiary" ? 20 : event.direction === "negative" ? -30 : 0) + (Date.parse(asOf) - Date.parse(event.publishedAt) < 864e5 ? 10 : 5)))) : null,
		events: events.map((event) => ({
			id: event.id,
			title: event.title,
			eventType: event.eventType,
			direct: event.direct,
			direction: event.direction,
			sourceUrl: event.sourceUrl
		})),
		negative: events.some((event) => event.direction === "negative")
	};
}
function riskSnapshot(snapshot, news, config) {
	const penalties = [], rules = config.risk;
	const add = (condition, points, reason) => {
		if (condition) penalties.push({
			points,
			reason
		});
	};
	add(snapshot.ma20Distance > rules.maximumMa20Distance, 25, "遠離20日均線");
	add(snapshot.returns[5] > rules.maximumFiveDayReturn, 25, "五日漲幅過熱");
	add(snapshot.returns[10] > rules.maximumTenDayReturn, 20, "十日漲幅過大");
	add(snapshot.rsi14 > rules.maximumRsi, 25, "RSI過熱");
	add(snapshot.gapPct > rules.maximumGapPct, 25, "跳空幅度過大");
	add(snapshot.atrPct > rules.maximumAtrPct, 15, "波動過大");
	add(snapshot.upperWickRatio > rules.maximumUpperWickRatio && snapshot.volumeRatio20 > 2, 15, "爆量長上影");
	add(snapshot.last.close < snapshot.ma20 && snapshot.volumeRatio20 > 2, 20, "爆量跌破20日均線");
	add(news.events.length > 0 && snapshot.returns[10] > 20, 15, "事件前已明顯上漲");
	add(news.negative, 15, "近期負向事件");
	return {
		penalty: Math.min(100, penalties.reduce((sum, item) => sum + item.points, 0)),
		items: penalties,
		overheated: penalties.some((item) => item.points >= 25)
	};
}
function setupSnapshot(snapshot, rs, news, config) {
	const pull = config.pullback, trend = snapshot.last.close > snapshot.ma20 && snapshot.ma20 > snapshot.ma60;
	const pullback = trend && snapshot.pullbackDays >= pull.minimumDays && snapshot.pullbackDays <= pull.maximumDays && snapshot.drawdown >= pull.minimumDrawdown && snapshot.drawdown <= pull.maximumDrawdown && snapshot.rsi14 >= pull.minimumRsi && snapshot.rsi14 <= pull.maximumRsi && snapshot.ma20Distance <= pull.maximumMa20Distance && finite(snapshot.pullbackVolume) && finite(snapshot.advanceVolume) && snapshot.pullbackVolume < snapshot.advanceVolume;
	const breakout = trend && snapshot.last.close > snapshot.previousHigh20 && snapshot.volumeRatio20 >= config.breakout.minimumVolumeRatio && snapshot.ma20Distance <= config.breakout.maximumMa20Distance && snapshot.rsi14 <= 75;
	const momentum = trend && rs >= 80 && snapshot.positiveCloseRatio >= .6 && snapshot.returns[20] > 0 && snapshot.returns[5] <= 12 && snapshot.rsi14 <= 70;
	const event = trend && news.score >= 70 && news.events.some((item) => item.direct && item.direction === "beneficiary") && snapshot.returns[10] <= 15;
	const types = [
		pullback ? "PULLBACK" : null,
		breakout ? "BREAKOUT" : null,
		momentum ? "MOMENTUM" : null,
		event ? "EVENT" : null
	].filter(Boolean);
	const score = pullback ? clamp(70 + (1 - snapshot.pullbackVolume / snapshot.advanceVolume) * 20 + (snapshot.ma20Distance <= 2 ? 10 : 0)) : breakout ? clamp(65 + Math.min(2, snapshot.volumeRatio20) * 10 + (snapshot.last.close > snapshot.previousHigh55 ? 15 : 0)) : momentum ? 75 : event ? 70 : 0;
	return {
		types,
		primary: types[0] || null,
		score
	};
}
function scanMarket(input, config = strategyConfig) {
	validateConfig(config);
	const { asOf, universe = [], benchmark = [], news = [], validation = null } = input;
	if (!finite(Date.parse(asOf))) throw Error("INVALID_AS_OF");
	const regime = marketRegime(benchmark, asOf, input.context || {}, config), reference = technicalSnapshot(benchmark, asOf), all = [];
	const excluded = {};
	for (const stock of universe) {
		const snapshot = technicalSnapshot(stock.bars || [], asOf), eligibility = eligibleSecurity(stock, snapshot, asOf, regime.date, config);
		if (!eligibility.eligible) {
			for (const reason of eligibility.reasons) excluded[reason] = (excluded[reason] || 0) + 1;
			continue;
		}
		all.push({
			stock,
			snapshot,
			eligibility
		});
	}
	const percentile = (value, values) => values.length > 1 ? 100 * (values.filter((other) => other < value).length + .5 * (values.filter((other) => other === value).length - 1)) / (values.length - 1) : 50;
	const rankPopulation = Array.isArray(input.rankUniverse) ? input.rankUniverse : all.map((item) => ({ returns: item.snapshot.returns }));
	const rankValues = Object.fromEntries([
		20,
		60,
		120
	].map((days) => [days, rankPopulation.map((item) => item.returns[days] - (reference?.returns[days] || 0)).filter(finite)]));
	const candidates = all.map(({ stock, snapshot, eligibility }) => {
		const rs = reference && [
			20,
			60,
			120
		].every((days) => finite(reference.returns[days])) ? [
			20,
			60,
			120
		].reduce((sum, days, index) => sum + percentile(snapshot.returns[days] - reference.returns[days], rankValues[days]) * [
			.3,
			.4,
			.3
		][index], 0) : null;
		const institutional = institutionalSnapshot(stock.institutional || [], asOf, snapshot.bars.map((bar) => bar.date), snapshot.avg20Volume), fundamental = fundamentalSnapshot(stock.revenue || [], asOf), catalyst = newsSnapshot(news, stock, asOf), risk = riskSnapshot(snapshot, catalyst, config), setup = setupSnapshot(snapshot, rs, catalyst, config);
		const trendScore = (snapshot.last.close > snapshot.ma20 ? 25 : 0) + (snapshot.ma20 > snapshot.ma60 ? 25 : 0) + (snapshot.ma60Slope > 0 ? 15 : 0) + (snapshot.last.close > snapshot.ma120 ? 15 : 0) + (snapshot.ma20Slope > 0 ? 10 : 0) + (snapshot.adx14 > 20 ? 10 : 0);
		const persistence = clamp(snapshot.positiveCloseRatio * 70 + (snapshot.averageUp > Math.abs(snapshot.averageDown || 0) ? 15 : 0) + (snapshot.returns[60] > 0 ? 15 : 0));
		const volumeScore = setup.primary === "PULLBACK" ? clamp(100 * (1 - snapshot.pullbackVolume / snapshot.advanceVolume) + 50) : setup.primary === "BREAKOUT" ? clamp(snapshot.volumeRatio20 * 40) : clamp(50 + (snapshot.priceVolumeCorrelation || 0) * 30);
		const scores = {
			relativeStrength: rs,
			trend: trendScore,
			setup: setup.score,
			momentumPersistence: persistence,
			volume: volumeScore,
			institutional: institutional.score,
			fundamental: fundamental.score,
			newsCatalyst: catalyst.score
		};
		const base = Object.entries(config.weights).reduce((sum, [key, weight]) => sum + (finite(scores[key]) ? scores[key] * weight / 100 : 0), 0), deterioration = (input.strategyHealth || {})[setup.primary], multiplier = deterioration?.samples >= config.deterioration.minimumSamples && finite(deterioration.profitFactor) && deterioration.profitFactor < 1 ? config.deterioration.multiplier : 1;
		const score = round(clamp((base - risk.penalty) * multiplier)), proof = validation?.strategies?.[setup.primary];
		const oosReady = validation?.status === "VALIDATED" && validation?.configVersion === config.version && validation.configIdentity === JSON.stringify(config) && Date.parse(validation.availableAt) <= Date.parse(asOf) && proof?.sampleSize >= config.minimumOosSamples && proof.profitFactor > 1 && proof.averageReturn > 0;
		const state = setup.primary && finite(rs) && regime.verified && !risk.overheated && score >= regime.threshold && oosReady && !eligibility.warnings.length && input.coverage?.complete === true && input.preOpen?.verified === true && Date.parse(input.preOpen.availableAt) <= Date.parse(asOf) ? "HIGH" : setup.primary && finite(rs) && !risk.overheated && score >= regime.threshold - config.watchDiscount ? "WATCH" : "SKIP";
		const last = snapshot.last, global = (input.globalContext || []).filter((item) => item.industries?.includes(stock.industry) && Date.parse(item.availableAt) <= Date.parse(asOf));
		return {
			symbol: stock.symbol,
			name: stock.name,
			market: stock.market,
			industry: stock.industry,
			state,
			primarySetup: setup.primary,
			setups: setup.types,
			score,
			scores: Object.fromEntries(Object.entries(scores).map(([key, value]) => [key, round(value)])),
			riskPenalty: risk.penalty,
			risk: risk.items,
			relativeStrengthScore: round(rs),
			returns: Object.fromEntries(Object.entries(snapshot.returns).map(([key, value]) => [key, round(value)])),
			vsTaiex: Object.fromEntries([
				5,
				10,
				20,
				60,
				120
			].map((days) => [days, finite(reference?.returns[days]) ? round(snapshot.returns[days] - reference.returns[days]) : null])),
			volumeRatio5: round(snapshot.volumeRatio5),
			priceVolumeCorrelation: round(snapshot.priceVolumeCorrelation),
			turnoverRate: stock.shares > 0 ? round(snapshot.last.volume / stock.shares * 100) : null,
			referencePrice: last.close,
			sourceDate: last.date,
			ma20: round(snapshot.ma20),
			ma60: round(snapshot.ma60),
			rsi14: round(snapshot.rsi14),
			adx14: round(snapshot.adx14),
			atr14: round(snapshot.atr14),
			volumeRatio20: round(snapshot.volumeRatio20),
			avg20Volume: round(snapshot.avg20Volume),
			avg20Turnover: round(snapshot.avg20Turnover),
			keyRange: [round(snapshot.ma20 - snapshot.atr14 * .5), round(snapshot.ma20 + snapshot.atr14 * .5)],
			riskReference: round(last.close - config.exits.atrMultiple * snapshot.atr14),
			institutional,
			fundamental,
			news: catalyst,
			globalContext: global,
			globalContextScore: global.length ? round(mean(global.map((item) => item.score).filter(finite))) : null,
			reasons: [
				snapshot.ma20 > snapshot.ma60 ? "MA20 高於 MA60" : null,
				finite(rs) ? `中期相對強度 ${round(rs)}` : null,
				setup.primary === "PULLBACK" ? "回檔量縮、接近20日均線" : null,
				setup.primary === "BREAKOUT" ? "突破前20日高點且量增" : null,
				institutional.trustStreak ? `投信連買 ${institutional.trustStreak} 個交易日` : null
			].filter(Boolean),
			gaps: [
				...eligibility.warnings,
				...!oosReady ? ["歷史樣本不足／尚無有效OOS驗證"] : [],
				...!institutional.complete ? ["五日法人資料不足"] : [],
				...!fundamental.complete ? ["三個月營收資料不足"] : [],
				...!input.preOpen?.verified ? ["未取得盤前試撮確認"] : [],
				...!global.length ? ["隔夜對應產業資料不足"] : []
			],
			strategyDeteriorated: multiplier < 1,
			validation: proof || null
		};
	}).sort((left, right) => right.score - left.score || left.market.localeCompare(right.market) || left.symbol.localeCompare(right.symbol));
	return {
		schema: 1,
		mode: "RESEARCH",
		configVersion: config.version,
		asOf,
		regime,
		universeSize: universe.length,
		eligibleUniverse: all.map(({ stock, snapshot }) => ({
			symbol: stock.symbol,
			name: stock.name,
			market: stock.market,
			avg20Volume: round(snapshot.avg20Volume),
			avg20Turnover: round(snapshot.avg20Turnover),
			lastPrice: snapshot.last.close
		})),
		eligibleCount: all.length,
		excluded,
		high: candidates.filter((stock) => stock.state === "HIGH").slice(0, config.maximumPicks),
		watch: candidates.filter((stock) => stock.state === "WATCH").slice(0, config.maximumWatch),
		candidates,
		validationStatus: validation?.status || "ACCUMULATING",
		coverage: input.coverage || { complete: false },
		preOpenVerified: input.preOpen?.verified === true
	};
}

//#endregion
//#region src/radar/backtest.js
function executePaperTrade(signal, rows, { horizon = 10, exit = "fixed", evaluationAt = (/* @__PURE__ */ new Date()).toISOString(), costs = TradingCostConfig, config = strategyConfig, corporateActions = [] } = {}) {
	if (!config.holdingHorizons.includes(horizon) || ![
		"fixed",
		"atr",
		"trailing",
		"ma20",
		"target"
	].includes(exit)) throw Error("INVALID_EXIT");
	if (![
		costs.notional,
		costs.commissionRate,
		costs.minimumCommission,
		costs.sellTaxRate,
		costs.slippageBps,
		costs.maximumVolumeParticipation
	].every(finite) || costs.notional <= 0 || costs.commissionRate < 0 || costs.minimumCommission < 0 || costs.sellTaxRate <= 0 || costs.slippageBps < 0 || costs.maximumVolumeParticipation <= 0) throw Error("INVALID_COSTS");
	const all = visibleRows(rows, evaluationAt).filter(validBar);
	const future = all.filter((bar) => bar.date > signal.sourceDate), entry = future[0];
	if (!entry) return {
		status: "PENDING",
		reason: "尚無下一交易日開盤"
	};
	if (entry.volume <= 0 || entry.high === entry.low && Math.abs(entry.open / signal.referencePrice - 1) >= .095) return {
		status: "UNFILLED",
		reason: "無法合理成交／一價漲跌停"
	};
	if ((entry.open / signal.referencePrice - 1) * 100 > config.risk.maximumGapPct) return {
		status: "UNFILLED",
		reason: "實際開盤跳空過大"
	};
	if (!finite(signal.avg20Volume) || signal.avg20Volume <= 0) return {
		status: "UNVERIFIED",
		reason: "訊號當時的平均成交量不足"
	};
	const slippage = costs.slippageBps / 1e4, entryPrice = entry.open * (1 + slippage), quantity = Math.min(Math.floor(costs.notional / (entryPrice * (1 + costs.commissionRate) + costs.minimumCommission)), Math.floor(signal.avg20Volume * costs.maximumVolumeParticipation));
	if (quantity < 1) return {
		status: "UNFILLED",
		reason: "資金或成交量不足"
	};
	if (quantity > entry.volume * costs.maximumVolumeParticipation) return {
		status: "UNFILLED",
		reason: "實際成交量不足以支持紙上成交假設"
	};
	const fee = (amount) => Math.max(costs.minimumCommission, amount * costs.commissionRate), entryValue = entryPrice * quantity, buyFee = fee(entryValue), entryCost = entryValue + buyFee;
	const maximum = Math.min(horizon, config.exits.maximumHoldingDays), atr = signal.atr14;
	if (exit !== "fixed" && (!finite(atr) || atr <= 0)) return {
		status: "UNFILLED",
		reason: "ATR資料不足"
	};
	let stop = entryPrice - config.exits.atrMultiple * atr, highest = entryPrice, exitBar = null, exitRaw = null, reason = "fixed", ambiguous = false, held = 0;
	for (let index = 0; index < Math.min(maximum, future.length); index++) {
		const bar = future[index];
		held = index + 1;
		if (corporateActions.some((action) => action.eventDate >= entry.date && action.eventDate <= bar.date)) return {
			status: "UNVERIFIED",
			reason: "持有期間有公司行動，須經還原／配息模型核實"
		};
		if (index > 0 && exit === "ma20") {
			const previousIndex = all.findIndex((point) => point.date === future[index - 1].date), previous = all[previousIndex], ma20 = previousIndex >= 19 ? mean(all.slice(previousIndex - 19, previousIndex + 1).map((point) => point.close)) : null;
			if (ma20 && previous.close < ma20) {
				exitBar = bar;
				exitRaw = bar.open;
				reason = "ma20_next_open";
				break;
			}
		}
		if ([
			"atr",
			"trailing",
			"target"
		].includes(exit)) {
			const target = entryPrice + config.exits.profitAtrMultiple * atr, hitStop = bar.low <= stop, hitTarget = exit === "target" && bar.high >= target;
			if (hitStop) {
				exitBar = bar;
				exitRaw = Math.min(bar.open, stop);
				reason = "atr_stop";
				ambiguous = hitTarget;
				break;
			}
			if (hitTarget) {
				exitBar = bar;
				exitRaw = Math.max(bar.open, target);
				reason = "profit_target";
				break;
			}
			highest = Math.max(highest, bar.high);
			if (exit === "trailing") stop = Math.max(stop, highest - config.exits.trailingAtrMultiple * atr);
		}
		if (index === maximum - 1) {
			exitBar = bar;
			exitRaw = bar.close;
			break;
		}
	}
	if (!exitBar) return {
		status: "PENDING",
		entryDate: entry.date,
		entryPrice: round(entryPrice),
		quantity,
		reason: "尚未到期"
	};
	if (exitBar.volume <= 0 || exitBar.high === exitBar.low && exitBar.close < (future[Math.max(0, held - 2)]?.close || entryPrice) * .905) return {
		status: "UNFILLED_EXIT",
		reason: "出場一價跌停／無量，不假定能成交"
	};
	const exitPrice = exitRaw * (1 - slippage), exitValue = exitPrice * quantity, sellFee = fee(exitValue), tax = exitValue * costs.sellTaxRate, proceeds = exitValue - sellFee - tax, pnl = proceeds - entryCost;
	return {
		status: "COMPLETE",
		symbol: signal.symbol,
		market: signal.market,
		strategy: signal.primarySetup,
		signalAt: signal.asOf,
		sourceDate: signal.sourceDate,
		entryDate: entry.date,
		exitDate: exitBar.date,
		availableAt: exitBar.availableAt,
		horizon,
		exitModel: exit,
		entryPrice: round(entryPrice),
		exitPrice: round(exitPrice),
		quantity,
		entryCost: round(entryCost),
		proceeds: round(proceeds),
		fees: round(buyFee + sellFee),
		tax: round(tax),
		slippageBps: costs.slippageBps,
		pnl: round(pnl),
		netReturn: pnl / entryCost,
		holdingDays: held,
		exitReason: reason,
		ambiguousIntraday: ambiguous
	};
}

//#endregion
//#region src/intelligence/engine.js
const themes = [
	[
		"memory",
		"記憶體",
		[
			"DRAM",
			"NAND",
			"NOR",
			"記憶體",
			"memory chip"
		],
		["Memory"],
		[
			"HBM",
			"DRAM",
			"NAND",
			"NOR",
			"Controller",
			"Module",
			"Testing",
			"Equipment",
			"Materials"
		]
	],
	[
		"optical",
		"光通訊",
		[
			"光通訊",
			"矽光子",
			"CPO",
			"silicon photonics",
			"optical transceiver"
		],
		["Networking"],
		["Optical Communication"]
	],
	[
		"plastics",
		"塑膠",
		[
			"塑膠",
			"塑化",
			"PVC",
			"polymer",
			"plastics"
		],
		["Materials"],
		["Plastics"]
	],
	[
		"chemicals",
		"化工",
		[
			"化工",
			"化學材料",
			"特用化學",
			"chemical"
		],
		["Materials"],
		["Chemicals"]
	],
	[
		"pcb",
		"PCB",
		[
			"PCB",
			"印刷電路板",
			"載板"
		],
		["Electronics"],
		["PCB"]
	],
	[
		"shipping",
		"航運",
		[
			"航運",
			"運價",
			"shipping",
			"freight"
		],
		["Transportation"],
		["Shipping"]
	],
	[
		"steel",
		"鋼鐵",
		[
			"鋼鐵",
			"鋼價",
			"steel"
		],
		["Materials"],
		["Steel"]
	],
	[
		"healthcare",
		"生技醫療",
		[
			"生技",
			"新藥",
			"醫療",
			"clinical trial"
		],
		["Healthcare"],
		["Biotechnology"]
	],
	[
		"ai",
		"AI Infrastructure",
		[
			"AI",
			"artificial intelligence",
			"人工智慧",
			"data center"
		],
		[
			"Semiconductor",
			"Server Hardware",
			"Networking",
			"Power",
			"Cooling"
		],
		[
			"GPU",
			"Foundry",
			"Advanced Packaging",
			"HBM",
			"Server ODM",
			"PCB",
			"Power Supply",
			"Thermal Management"
		]
	],
	[
		"semiconductor",
		"Semiconductor",
		[
			"semiconductor",
			"chip",
			"半導體"
		],
		["Semiconductor"],
		[
			"Foundry",
			"GPU",
			"Equipment",
			"Advanced Packaging"
		]
	],
	[
		"hbm",
		"HBM",
		["HBM", "高頻寬記憶體"],
		["Memory"],
		["HBM"]
	],
	[
		"packaging",
		"Advanced Packaging",
		[
			"CoWoS",
			"advanced packaging",
			"先進封裝"
		],
		["Semiconductor"],
		["Advanced Packaging"]
	],
	[
		"server",
		"AI Server",
		["AI server", "AI伺服器"],
		["Server Hardware"],
		["Server ODM"]
	],
	[
		"cloud",
		"Cloud",
		[
			"cloud computing",
			"雲端",
			"hyperscaler"
		],
		["Cloud"],
		["Cloud Platform"]
	],
	[
		"power",
		"Power",
		[
			"electricity",
			"power grid",
			"電網",
			"電力"
		],
		["Power"],
		["Power Supply", "Grid"]
	],
	[
		"cooling",
		"Cooling",
		[
			"cooling",
			"liquid cooling",
			"散熱",
			"液冷"
		],
		["Cooling"],
		["Thermal Management"]
	],
	[
		"defense",
		"Defense",
		[
			"defense",
			"military",
			"國防",
			"軍工"
		],
		["Defense"],
		["Defense Systems"]
	],
	[
		"oil",
		"Oil",
		[
			"oil",
			"OPEC",
			"原油"
		],
		["Energy"],
		["Oil Production", "Refining"]
	],
	[
		"rates",
		"Rates",
		[
			"interest rate",
			"Federal Reserve",
			"Fed",
			"利率",
			"央行"
		],
		[
			"Banks",
			"Financial Services",
			"Real Estate"
		],
		["Banking", "Lending"]
	],
	[
		"banks",
		"Banks",
		[
			"bank",
			"banking",
			"銀行"
		],
		["Banks", "Financial Services"],
		["Banking"]
	],
	[
		"ev",
		"EV",
		["electric vehicle", "電動車"],
		["Automotive"],
		["EV", "Battery"]
	],
	[
		"automation",
		"Industrial Automation",
		[
			"automation",
			"robotics",
			"自動化",
			"機器人"
		],
		["Industrial Automation"],
		["Robotics", "Industrial Computing"]
	],
	[
		"dollar",
		"美元",
		["dollar", "美元"],
		["Financial Services"],
		["Currency"]
	],
	[
		"inflation",
		"通膨",
		["inflation", "通膨"],
		["Consumer", "Financial Services"],
		["Pricing", "Lending"]
	]
].map(([id, name, keywords, industries, valueChainSegments]) => ({
	id,
	name,
	keywords,
	industries,
	valueChainSegments
}));
function safeUrl(value) {
	try {
		const url = new URL(value);
		return url.protocol === "https:" && !url.username && !url.password && !url.port && !/^(localhost|\[|\d+\.\d+\.\d+\.\d+$)/.test(url.hostname) && !url.hostname.endsWith(".local") ? url.href : null;
	} catch {
		return null;
	}
}
const text = (value) => String(value || "").normalize("NFKC").toLowerCase();
const symbol = (value) => String(value || "").toUpperCase().replace(/\.(TW|TWO)$/, "");
const escapePattern = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
function companyMentions(cluster, stock) {
	const names = [stock.name, ...stock.aliases || []].filter((name) => typeof name === "string" && (name.length >= 3 || /^[\u4e00-\u9fff]{2}$/.test(name)));
	return (cluster.sources || []).flatMap((source) => {
		const matches = [
			["標題", source.headline || source.title],
			["摘要", source.description],
			["摘要", source.excerpt]
		].flatMap(([field, value]) => {
			const content = text(value).replace(/<[^>]*>/g, " ").replace(/&#(?:\d+|x[\da-f]+);|&[a-z]+;/gi, " ");
			const name = names.find((name) => {
				const normalized = text(name), pattern = escapePattern(normalized);
				if (!content.includes(normalized)) return false;
				if (/^[\u4e00-\u9fff]{2}$/.test(name)) return new RegExp(String.raw`(?:^|[^\p{L}\p{N}]|看好|提及|包括)${pattern}(?=$|[^\p{L}\p{N}]|公司|股價|股票|營收|財報|投資|公告|合作|等|與|及)`, "u").test(content) || ![
					"大量",
					"世界",
					"統一",
					"中興",
					"中環",
					"全新",
					"新興",
					"中華"
				].includes(name) && new RegExp(pattern + "(?:公司|股價|股票|營收|財報|投資|公告|合作)").test(content);
				if (/^[a-z\d .&'-]+$/i.test(name)) return new RegExp("(?:^|[^a-zd])" + pattern + "(?=$|[^a-zd])", "u").test(content);
				return content.includes(normalized);
			});
			const ticker = content.includes(text(stock.symbol)) && new RegExp("(?:\\$|(?:^|[^a-z0-9])" + escapePattern(text(stock.market)) + ":)" + escapePattern(text(stock.symbol)) + "(?=$|[^a-z0-9])").test(content);
			return name || ticker ? [{
				url: safeUrl(source.url),
				field,
				name: name || stock.symbol
			}] : [];
		});
		if ((source.stockSymbols || []).some((value) => symbol(value) === stock.symbol)) matches.push({
			url: safeUrl(source.url),
			field: "股票代碼",
			name: stock.symbol
		});
		return [...new Map(matches.filter((match) => match.url).map((match) => [match.url + match.field, match])).values()];
	});
}

//#endregion
//#region src/intelligence/stock-market.js
function validStockDate(value, now = Date.now()) {
	return typeof value === "string" && /^\d{4}-\d{2}-\d{2}$/.test(value) && Number.isFinite(Date.parse(value)) && new Date(value).toISOString().slice(0, 10) === value && Date.parse(value) <= now;
}
function normalizeInstitutional(rows, symbol, now = Date.now()) {
	if (!Array.isArray(rows) || rows.length > 1e3 || !rows.length) throw Error("INSTITUTIONAL_UNAVAILABLE");
	const days = /* @__PURE__ */ new Map();
	for (const row of rows) {
		if (row.stock_id !== symbol || !validStockDate(row.date, now) || !Number.isSafeInteger(row.buy) || !Number.isSafeInteger(row.sell) || row.buy < 0 || row.sell < 0) throw Error("INVALID_INSTITUTIONAL");
		const values = days.get(row.date) || /* @__PURE__ */ new Map();
		if (values.has(row.name)) throw Error("DUPLICATE_INSTITUTIONAL");
		values.set(row.name, row.buy - row.sell);
		days.set(row.date, values);
	}
	return [...days].sort(([left], [right]) => left.localeCompare(right)).slice(-30).map(([date, values]) => {
		const sum = (names) => names.every((name) => values.has(name)) ? names.reduce((total, name) => total + values.get(name), 0) : null;
		const foreign = sum(["Foreign_Investor", "Foreign_Dealer_Self"]), trust = sum(["Investment_Trust"]);
		const dealer = values.has("Dealer_self") || values.has("Dealer_Hedging") ? sum(["Dealer_self", "Dealer_Hedging"]) : sum(["Dealer"]);
		return {
			date,
			foreign,
			trust,
			dealer,
			total: [
				foreign,
				trust,
				dealer
			].every(Number.isFinite) ? foreign + trust + dealer : null
		};
	});
}

//#endregion
//#region cloudflare/radar/sources.mjs
const radarSources = {
	"members:TWSE": "https://openapi.twse.com.tw/v1/opendata/t187ap03_L",
	"members:TPEx": "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O",
	"prices:TWSE": "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",
	"prices:TPEx": "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes",
	"institutional:TWSE": "https://www.twse.com.tw/rwd/zh/fund/T86?response=json&selectType=ALLBUT0999",
	"institutional:TPEx": "https://www.tpex.org.tw/openapi/v1/tpex_3insti_daily_trading",
	"revenue:TWSE": "https://openapi.twse.com.tw/v1/opendata/t187ap05_L",
	"revenue:TPEx": "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap05_O",
	"risk:TWSE": "https://openapi.twse.com.tw/v1/announcement/punish",
	"risk:TPEx": "https://www.tpex.org.tw/openapi/v1/tpex_disposal_information",
	news: "https://holdings-intelligence.b28228250.workers.dev/api/intelligence/latest"
};
const taipeiDay = (now) => new Intl.DateTimeFormat("en-CA", {
	timeZone: "Asia/Taipei",
	year: "numeric",
	month: "2-digit",
	day: "2-digit"
}).format(new Date(now));
function sourceNumber(value) {
	if (typeof value === "number") return Number.isFinite(value) ? value : null;
	if (typeof value !== "string" || !value.trim() || !/^[+-]?[\d,]+(?:\.\d+)?$/.test(value.trim())) return null;
	const result = Number(value.replaceAll(",", "").trim());
	return Number.isFinite(result) ? result : null;
}
function sourceDate(value) {
	const compact = String(value || "").replace(/[^\d]/g, "");
	let year, month, day;
	if (compact.length === 7) {
		year = Number(compact.slice(0, 3)) + 1911;
		month = compact.slice(3, 5);
		day = compact.slice(5, 7);
	} else if (compact.length === 8) {
		year = Number(compact.slice(0, 4));
		month = compact.slice(4, 6);
		day = compact.slice(6, 8);
	} else return null;
	const date = `${year}-${month}-${day}`;
	return Number.isFinite(Date.parse(date)) && new Date(date).toISOString().slice(0, 10) === date ? date : null;
}
function normalizeRadarSource(dataset, value, now = Date.now()) {
	const availableAt = new Date(now).toISOString(), market = dataset.split(":")[1], kind = dataset.split(":")[0], today = taipeiDay(now);
	let rows = [];
	if (kind === "news") {
		if (value.schema !== 1 || !Array.isArray(value.stories) || !Number.isFinite(Date.parse(value.generatedAt))) throw Error("INVALID_NEWS");
		return {
			dataset,
			sourceDate: today,
			availableAt,
			sourceUrl: radarSources[dataset],
			rows: [...new Map([...value.stories, ...value.categoryStories || []].map((story) => [story.id, story])).values()].map((story) => ({
				id: story.id,
				title: story.titleZh || story.headline,
				publishedAt: story.publishedAt,
				availableAt,
				eventType: story.primaryEventType,
				importance: story.importance,
				sources: (story.sources || []).map((source) => ({
					headline: source.headline || source.title,
					description: source.description,
					excerpt: source.excerpt,
					url: source.url,
					stockSymbols: source.stockSymbols || []
				})),
				direction: "uncertain",
				companyDirections: (story.relatedStocks || []).filter((stock) => [
					"beneficiary",
					"negative",
					"mixed",
					"neutral"
				].includes(stock.eventDirection) && stock.directionBasis?.some((item) => item.reason && story.sources.some((source) => source.url === item.sourceUrl))).map((stock) => ({
					symbol: stock.symbol,
					market: stock.market,
					direction: stock.eventDirection
				}))
			}))
		};
	}
	if (kind === "institutional" && market === "TWSE") {
		if (!Array.isArray(value.fields) || !Array.isArray(value.data) || !sourceDate(value.date)) throw Error("INVALID_INSTITUTIONAL");
		const index = (name) => value.fields.indexOf(name), net = (row, name) => sourceNumber(row[index(name)]), date = sourceDate(value.date);
		const required = [
			"證券代號",
			"外陸資買賣超股數(不含外資自營商)",
			"外資自營商買賣超股數",
			"投信買賣超股數",
			"自營商買賣超股數"
		];
		if (required.some((name) => index(name) < 0)) throw Error("INVALID_INSTITUTIONAL_FIELDS");
		rows = value.data.filter((row) => /^[1-9]\d{3}$/.test(row[0])).map((row) => {
			const foreign = net(row, required[1]), foreignDealer = net(row, required[2]);
			return {
				market,
				symbol: row[index("證券代號")],
				date,
				availableAt,
				foreign: foreign != null && foreignDealer != null ? foreign + foreignDealer : null,
				trust: net(row, required[3]),
				dealer: net(row, required[4])
			};
		});
	} else {
		if (!Array.isArray(value) || value.length > 3e4) throw Error("INVALID_SOURCE");
		for (const raw of value) {
			const row = Object.fromEntries(Object.entries(raw).map(([key, item]) => [key.trim(), item])), symbol = String(row["公司代號"] || row.Code || row.SecuritiesCompanyCode || "");
			if (!/^[1-9]\d{3}$/.test(symbol)) continue;
			const date = sourceDate(row["出表日期"] || row.Date) || today;
			if (date > today) throw Error("FUTURE_SOURCE");
			if (kind === "members") rows.push({
				symbol,
				market,
				name: row["公司簡稱"] || row.CompanyAbbreviation,
				industry: String(row["產業別"] || row.SecuritiesIndustryCode || ""),
				shares: sourceNumber(row["已發行普通股數或TDR原股發行股數"] || row.IssueShares),
				listedAt: sourceDate(row["上市日期"] || row.DateOfListing),
				securityType: "COMMON",
				availableAt,
				date
			});
			if (kind === "prices") {
				const point = {
					symbol,
					market,
					date,
					availableAt,
					open: sourceNumber(row.OpeningPrice || row.Open),
					high: sourceNumber(row.HighestPrice || row.High),
					low: sourceNumber(row.LowestPrice || row.Low),
					close: sourceNumber(row.ClosingPrice || row.Close),
					volume: sourceNumber(row.TradeVolume || row.TradingShares),
					turnover: sourceNumber(row.TradeValue || row.TransactionAmount),
					change: sourceNumber(row.Change),
					bid: sourceNumber(row.LatestBidPrice),
					ask: sourceNumber(row.LatesAskPrice || row.LatestAskPrice)
				};
				if (validBar(point)) rows.push(point);
			}
			if (kind === "institutional") rows.push({
				symbol,
				market,
				date,
				availableAt,
				foreign: sourceNumber(row["ForeignInvestorsInclude MainlandAreaInvestors-Difference"] ?? row["ForeignInvestorsIncludeMainlandAreaInvestors-Difference"]),
				trust: sourceNumber(row["SecuritiesInvestmentTrustCompanies-Difference"]),
				dealer: sourceNumber(row["Dealers-Difference"])
			});
			if (kind === "revenue") rows.push({
				symbol,
				market,
				date,
				availableAt,
				reportMonth: sourceDate(String(row["資料年月"]) + "01")?.slice(0, 7) || null,
				yoy: sourceNumber(row["營業收入-去年同月增減(%)"]),
				mom: sourceNumber(row["營業收入-上月比較增減(%)"]),
				revenue: sourceNumber(row["營業收入-當月營收"]) != null ? sourceNumber(row["營業收入-當月營收"]) * 1e3 : null,
				revenueUnit: "TWD"
			});
			if (kind === "risk") {
				const dates = String(row.DispositionPeriod || "").match(/\d{3}[\/]?\d{2}[\/]?\d{2}/g) || [];
				rows.push({
					symbol,
					market,
					date,
					availableAt,
					startDate: sourceDate(dates[0]),
					endDate: sourceDate(dates[1]),
					disposition: true
				});
			}
		}
	}
	if (kind !== "risk" && rows.length < 20) throw Error("SOURCE_COVERAGE_TOO_LOW");
	return {
		dataset,
		sourceDate: rows.map((row) => row.date).sort().at(-1) || today,
		availableAt,
		sourceUrl: radarSources[dataset],
		rows
	};
}
async function collectRadarSource(dataset, { fetcher = fetch, now = Date.now() } = {}) {
	if (!radarSources[dataset]) throw Error("UNKNOWN_DATASET");
	const original = new URL(radarSources[dataset]);
	let url = original.href, response;
	for (let attempt = 0; attempt < 3; attempt++) {
		response = await fetcher(url, {
			redirect: "manual",
			signal: AbortSignal.timeout(2e4),
			headers: { Accept: "application/json" }
		});
		if (![
			301,
			302,
			307,
			308
		].includes(response.status)) break;
		const location = response.headers.get("Location");
		if (!location) break;
		const next = new URL(location, url);
		if (next.protocol !== "https:" || next.hostname !== original.hostname || next.pathname !== original.pathname || next.username || next.password || next.port || /captcha|challenge|cf_chl/i.test(next.href)) break;
		url = next.href;
	}
	if (!response.ok && dataset === "members:TPEx") {
		const fallback = await fetcher("https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInfo", {
			redirect: "manual",
			signal: AbortSignal.timeout(2e4)
		});
		if (!fallback.ok) throw Error("SOURCE_HTTP_" + fallback.status);
		const value = await fallback.json();
		if (value.status !== 200 || !Array.isArray(value.data)) throw Error("INVALID_MEMBERS");
		const latest = /* @__PURE__ */ new Map();
		for (const row of value.data.filter((row) => /^[1-9]\d{3}$/.test(row.stock_id))) if (!latest.has(row.stock_id) || latest.get(row.stock_id).date < row.date) latest.set(row.stock_id, row);
		const availableAt = new Date(now).toISOString(), rows = [...latest.values()].filter((row) => row.type === "tpex" && Date.parse(row.date) <= now && now - Date.parse(row.date) < 6048e5).map((row) => ({
			market: "TPEx",
			symbol: row.stock_id,
			name: row.stock_name,
			industry: row.industry_category,
			listedAt: null,
			shares: null,
			securityType: "COMMON",
			date: row.date,
			availableAt
		}));
		if (rows.length < 500) throw Error("SOURCE_COVERAGE_TOO_LOW");
		return {
			dataset,
			sourceDate: taipeiDay(now),
			availableAt,
			sourceUrl: "https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInfo",
			provider: "FinMind",
			rows
		};
	}
	if (!response.ok) throw Error("SOURCE_HTTP_" + response.status);
	const body = await response.text();
	if (body.length > 18e6) throw Error("SOURCE_TOO_LARGE");
	let value;
	try {
		value = JSON.parse(body);
	} catch {
		throw Error("INVALID_JSON");
	}
	return normalizeRadarSource(dataset, value, now);
}
async function collectSeedHistory(market, symbol, { fetcher = fetch, now = Date.now(), recent = false } = {}) {
	if (!(["TWSE", "TPEx"].includes(market) && /^[1-9]\d{3}$/.test(symbol) || market === "INDEX" && symbol === "TAIEX")) throw Error("INVALID_SECURITY");
	const response = await fetcher("https://api.finmindtrade.com/api/v4/data?" + new URLSearchParams({
		dataset: "TaiwanStockPrice",
		data_id: symbol,
		start_date: (/* @__PURE__ */ new Date(now - (recent ? 14 : 420) * 864e5)).toISOString().slice(0, 10),
		end_date: taipeiDay(now)
	}), {
		redirect: "manual",
		signal: AbortSignal.timeout(15e3)
	});
	if (!response.ok) throw Error("HISTORY_HTTP_" + response.status);
	const value = await response.json();
	if (value.status !== 200 || !Array.isArray(value.data) || value.data.length > 400) throw Error("INVALID_HISTORY");
	const availableAt = new Date(now).toISOString(), bars = value.data.map((row) => ({
		date: row.date,
		availableAt,
		open: row.open,
		high: row.max,
		low: row.min,
		close: row.close,
		volume: row.Trading_Volume,
		turnover: row.Trading_money
	}));
	if (bars.some((bar) => !validBar(bar) || bar.date > taipeiDay(now)) || new Set(bars.map((bar) => bar.date)).size !== bars.length || !bars.length) throw Error("INVALID_HISTORY");
	return {
		market,
		symbol,
		availableAt,
		bars,
		provenance: "OBSERVED_NOW_NOT_HISTORICAL_PIT"
	};
}
async function collectSeedExtras(market, symbol, { fetcher = fetch, now = Date.now() } = {}) {
	const availableAt = new Date(now).toISOString();
	const get = async (dataset, days) => {
		const response = await fetcher("https://api.finmindtrade.com/api/v4/data?" + new URLSearchParams({
			dataset,
			data_id: symbol,
			start_date: (/* @__PURE__ */ new Date(now - days * 864e5)).toISOString().slice(0, 10),
			end_date: taipeiDay(now)
		}), {
			redirect: "manual",
			signal: AbortSignal.timeout(15e3)
		});
		if (!response.ok) throw Error();
		const value = await response.json();
		if (value.status !== 200 || !Array.isArray(value.data) || value.data.length > 1e3) throw Error();
		return value.data;
	};
	let institutional = [], revenue = [];
	try {
		institutional = normalizeInstitutional(await get("TaiwanStockInstitutionalInvestorsBuySell", 35), symbol, now).map((row) => ({
			...row,
			market,
			symbol,
			availableAt
		}));
	} catch {}
	try {
		const rows = await get("TaiwanStockMonthRevenue", 500), months = new Map(rows.filter((row) => row.stock_id === symbol && Number.isFinite(row.revenue) && row.revenue >= 0).map((row) => [Number(row.revenue_year) * 12 + Number(row.revenue_month), row]));
		revenue = [...months].sort(([left], [right]) => left - right).slice(-3).map(([month, row]) => {
			const previous = months.get(month - 1)?.revenue, year = months.get(month - 12)?.revenue;
			return {
				market,
				symbol,
				availableAt,
				date: sourceDate(String(row.create_time || "").slice(0, 10)) || row.date,
				reportMonth: `${row.revenue_year}-${String(row.revenue_month).padStart(2, "0")}`,
				revenue: row.revenue,
				yoy: year > 0 ? (row.revenue / year - 1) * 100 : null,
				mom: previous > 0 ? (row.revenue / previous - 1) * 100 : null
			};
		});
	} catch {}
	return {
		institutional,
		revenue
	};
}

//#endregion
//#region cloudflare/radar/repository.mjs
const dbRows = async (promise) => (await promise).results || [];
function factStatement(db, row, kind) {
	const payload = JSON.stringify(row);
	return db.prepare("INSERT OR IGNORE INTO radar_facts(market,symbol,kind,date,available_at,payload) SELECT ?,?,?,?,?,? WHERE COALESCE((SELECT json_remove(payload,'$.availableAt') FROM radar_facts WHERE market=? AND symbol=? AND kind=? AND date=? ORDER BY available_at DESC LIMIT 1),'')!=json_remove(?,'$.availableAt')").bind(row.market, row.symbol, kind, row.date, row.availableAt, payload, row.market, row.symbol, kind, row.date, payload);
}
async function storeSnapshot(db, snapshot) {
	const existing = await db.prepare("SELECT payload FROM radar_snapshots WHERE dataset=? AND source_date=? ORDER BY available_at DESC LIMIT 1").bind(snapshot.dataset, snapshot.sourceDate).first();
	const fingerprint = (value) => JSON.stringify(value.rows.map(({ availableAt, ...row }) => row));
	if (existing && fingerprint(JSON.parse(existing.payload)) === fingerprint(snapshot)) return false;
	const kind = snapshot.dataset.split(":")[0], statements = [db.prepare("INSERT OR IGNORE INTO radar_snapshots(dataset,source_date,available_at,source_url,payload) VALUES(?,?,?,?,?)").bind(snapshot.dataset, snapshot.sourceDate, snapshot.availableAt, snapshot.sourceUrl, JSON.stringify(snapshot))];
	if (kind === "members") statements.push(db.prepare("UPDATE radar_members SET active=0 WHERE market=?").bind(snapshot.dataset.split(":")[1]));
	for (const row of snapshot.rows) {
		if (kind === "members") statements.push(db.prepare("INSERT INTO radar_members(market,symbol,name,first_seen,payload,active) VALUES(?,?,?,?,?,1) ON CONFLICT(market,symbol) DO UPDATE SET name=excluded.name,payload=excluded.payload,active=1").bind(row.market, row.symbol, row.name, row.availableAt, JSON.stringify(row)));
		if (kind !== "news") statements.push(factStatement(db, row, kind));
	}
	return (await db.batch(statements))[0].meta.changes === 1;
}
async function latestSnapshot(db, dataset, asOf) {
	return db.prepare("SELECT payload FROM radar_snapshots WHERE dataset=? AND available_at<=? ORDER BY available_at DESC LIMIT 1").bind(dataset, asOf).first().then((row) => row ? JSON.parse(row.payload) : null);
}
async function currentMembers(db, asOf) {
	const members = [];
	for (const market of ["TWSE", "TPEx"]) {
		const snapshot = await latestSnapshot(db, "members:" + market, asOf);
		if (snapshot) members.push(...snapshot.rows);
	}
	return members;
}
async function stockInputs(db, keys, asOf) {
	const stocks = [];
	for (const key of keys) {
		const member = await db.prepare("SELECT first_seen,payload FROM radar_members WHERE market=? AND symbol=? AND first_seen<=?").bind(key.market, key.symbol, asOf).first();
		if (!member) continue;
		const history = await db.prepare("SELECT available_at,payload FROM radar_history WHERE market=? AND symbol=? AND available_at<=?").bind(key.market, key.symbol, asOf).first();
		const facts = await dbRows(db.prepare("SELECT kind,payload FROM radar_facts WHERE market=? AND symbol=? AND available_at<=? ORDER BY date").bind(key.market, key.symbol, asOf).all());
		const metadata = facts.filter((row) => row.kind === "members").map((row) => JSON.parse(row.payload)).at(-1) || JSON.parse(member.payload);
		if (Date.parse(metadata.availableAt) > Date.parse(asOf)) continue;
		const prices = facts.filter((row) => row.kind === "prices").map((row) => JSON.parse(row.payload)), last = prices.at(-1), riskSource = await db.prepare("SELECT available_at FROM radar_snapshots WHERE dataset=? AND available_at<=? ORDER BY available_at DESC LIMIT 1").bind("risk:" + key.market, asOf).first();
		const disposition = facts.filter((row) => row.kind === "risk").map((row) => JSON.parse(row.payload)).some((row) => row.startDate && row.endDate && row.startDate <= asOf.slice(0, 10) && row.endDate >= asOf.slice(0, 10));
		const spread = last?.ask > 0 && last?.bid > 0 && last.ask >= last.bid ? (last.ask - last.bid) / ((last.ask + last.bid) / 2) * 100 : null;
		const risk = [{
			date: asOf.slice(0, 10),
			availableAt: asOf,
			disposition,
			dispositionVerified: !!riskSource,
			spreadPct: spread
		}];
		stocks.push({
			...metadata,
			availableAt: member.first_seen,
			bars: [...history ? JSON.parse(history.payload).bars : [], ...prices],
			institutional: facts.filter((row) => row.kind === "institutional").map((row) => JSON.parse(row.payload)),
			revenue: facts.filter((row) => row.kind === "revenue").map((row) => JSON.parse(row.payload)),
			risk
		});
	}
	return stocks;
}
async function benchmarkInput(db, asOf) {
	const row = await db.prepare("SELECT payload FROM radar_history WHERE market='INDEX' AND symbol='TAIEX' AND available_at<=?").bind(asOf).first(), facts = await dbRows(db.prepare("SELECT payload FROM radar_facts WHERE market='INDEX' AND symbol='TAIEX' AND available_at<=? ORDER BY date").bind(asOf).all());
	return [...row ? JSON.parse(row.payload).bars : [], ...facts.map((item) => JSON.parse(item.payload))];
}
async function recordEvent(db, kind, code, now = Date.now()) {
	await db.prepare("INSERT INTO radar_events(created_at,kind,code) VALUES(?,?,?)").bind(new Date(now).toISOString(), kind, code).run();
}
async function enqueueBatch(env, jobs, now = Date.now()) {
	if (!jobs.length) return 0;
	const date = taipeiDay(now), limit = Number(env.DAILY_MESSAGE_LIMIT || 2250), used = await env.RADAR_DB.prepare("SELECT COUNT(*) AS count FROM radar_jobs WHERE id LIKE ?").bind(date + ":%").first();
	const selected = jobs.slice(0, Math.max(0, limit - used.count));
	if (!selected.length) return 0;
	const claimed = await env.RADAR_DB.batch(selected.map((job) => env.RADAR_DB.prepare("INSERT INTO radar_jobs(id,status,updated_at) VALUES(?,'queued',?) ON CONFLICT(id) DO UPDATE SET status='queued',updated_at=excluded.updated_at WHERE status='send_failed'").bind(job.id, new Date(now).toISOString())));
	const fresh = selected.filter((job, index) => claimed[index].meta.changes > 0);
	for (let index = 0; index < fresh.length; index += 100) {
		const group = fresh.slice(index, index + 100);
		try {
			await env.RADAR_JOBS.sendBatch(group.map((job) => ({ body: {
				...job.body,
				jobId: job.id
			} })));
		} catch (error) {
			await env.RADAR_DB.batch(group.map((job) => env.RADAR_DB.prepare("UPDATE radar_jobs SET status='send_failed' WHERE id=?").bind(job.id)));
			throw error;
		}
	}
	return fresh.length;
}
async function savePublication(db, snapshot, signals) {
	const publication = {
		...snapshot,
		paperSignalIds: signals.map((signal) => signal.id)
	}, statements = [db.prepare("INSERT OR IGNORE INTO radar_publications(session,run_id,published_at,payload) VALUES(?,?,?,?)").bind(snapshot.session, snapshot.runId, snapshot.publishedAt, JSON.stringify(publication))];
	for (const signal of signals) statements.push(db.prepare("INSERT OR IGNORE INTO radar_signals(id,session,market,symbol,strategy,score,reference_price,config_version,available_at,payload) SELECT ?,?,?,?,?,?,?,?,?,? WHERE EXISTS (SELECT 1 FROM radar_publications p,json_each(p.payload,'$.paperSignalIds') ids WHERE p.session=? AND ids.value=?)").bind(signal.id, snapshot.session, signal.market, signal.symbol, signal.primarySetup, signal.score, signal.referencePrice, snapshot.configVersion, snapshot.publishedAt, JSON.stringify(signal), snapshot.session, signal.id));
	await db.batch(statements);
}

//#endregion
//#region cloudflare/radar/prepared.mjs
const preparedDeadline = "05:40";
function validatePrepared(value, now = Date.now()) {
	const reject = () => {
		throw Error("INVALID_PREPARED_RESEARCH");
	};
	if (!value || value.schema !== 1 || value.mode !== "RESEARCH" || value.configVersion !== strategyConfig.version || value.status !== "READY") reject();
	if (!/^\d{4}-\d{2}-\d{2}$/.test(value.session) || taipeiDay(Date.parse(value.session + "T12:00:00+08:00")) !== value.session) reject();
	if ([0, 6].includes((/* @__PURE__ */ new Date(value.session + "T00:00:00Z")).getUTCDay())) reject();
	const cutoff = Date.parse(value.asOf), created = Date.parse(value.preparedAt), deadline = Date.parse(value.session + "T05:40:00+08:00");
	if (!Number.isFinite(cutoff) || !Number.isFinite(created) || cutoff > created || created > now + 3e4 || created >= deadline || cutoff < Date.parse(value.session + "T00:00:00+08:00") || now >= deadline) reject();
	if (value.runId !== value.session + ":" + strategyConfig.version || !Number.isInteger(value.universeSize) || value.universeSize < 1 || value.universeSize > 1e4 || value.processedCount !== value.universeSize || !Number.isInteger(value.eligibleCount) || value.eligibleCount < 0 || value.eligibleCount > value.universeSize) reject();
	if (value.high?.length !== 0 || !Array.isArray(value.watch) || value.watch.length > 20 || value.validationStatus !== "ACCUMULATING" || value.preOpenVerified !== false) reject();
	if (![
		"Bull",
		"Neutral",
		"Risk-Off"
	].includes(value.regime?.state) || value.regime.threshold !== strategyConfig.thresholds[value.regime.state]) reject();
	const keys = /* @__PURE__ */ new Set();
	for (const stock of value.watch) {
		const key = stock.market + ":" + stock.symbol;
		if (!["TWSE", "TPEx"].includes(stock.market) || !/^[1-9]\d{3}$/.test(stock.symbol) || typeof stock.name !== "string" || stock.name.length > 100 || stock.state !== "WATCH" || !Number.isFinite(stock.score) || stock.score < 0 || stock.score > 100 || keys.has(key)) reject();
		keys.add(key);
	}
	if (!Array.isArray(value.gaps) || value.gaps.some((gap) => typeof gap !== "string" || gap.length > 300)) reject();
	if (/"(?:password|otp|accessToken|apiKey|portfolio|transactions|cashBalances|deviceId)"\s*:/i.test(JSON.stringify(value))) reject();
	return value;
}

//#endregion
//#region cloudflare/radar/worker.mjs
const parse = (row) => JSON.parse(row.payload);
async function strategyHealth(db, asOf) {
	const rows = await dbRows(db.prepare("SELECT strategy,COUNT(*) AS samples,SUM(CASE WHEN pnl>0 THEN pnl ELSE 0 END) AS profit,SUM(CASE WHEN pnl<0 THEN -pnl ELSE 0 END) AS loss FROM (SELECT json_extract(o.payload,'$.strategy') AS strategy,json_extract(o.payload,'$.pnl') AS pnl,ROW_NUMBER() OVER(PARTITION BY json_extract(o.payload,'$.strategy') ORDER BY o.available_at DESC) AS recent FROM radar_outcomes o JOIN radar_signals s ON s.id=o.signal_id WHERE o.horizon=10 AND o.exit_model='atr' AND o.available_at<=? AND s.config_version=? AND json_extract(o.payload,'$.status')='COMPLETE') WHERE recent<=30 GROUP BY strategy").bind(asOf, strategyConfig.version).all());
	return Object.fromEntries(rows.map((row) => [row.strategy, {
		samples: row.samples,
		profitFactor: row.loss > 0 ? row.profit / row.loss : null
	}]));
}
async function sourceJob(env, dataset, now) {
	const db = env.RADAR_DB, stamp = new Date(now).toISOString();
	try {
		const snapshot = await collectRadarSource(dataset, {
			now,
			fetcher: dataset === "news" && env.INTELLIGENCE ? (url, options) => env.INTELLIGENCE.fetch(url, options) : fetch
		});
		await storeSnapshot(db, snapshot);
		await db.prepare("INSERT INTO radar_source_checks(dataset,checked_at,status,row_count,source_date) VALUES(?,?,?,?,?) ON CONFLICT(dataset) DO UPDATE SET checked_at=excluded.checked_at,status=excluded.status,row_count=excluded.row_count,source_date=excluded.source_date").bind(dataset, stamp, "success", snapshot.rows.length, snapshot.sourceDate).run();
	} catch (error) {
		await db.prepare("INSERT INTO radar_source_checks(dataset,checked_at,status) VALUES(?,?,?) ON CONFLICT(dataset) DO UPDATE SET checked_at=excluded.checked_at,status=excluded.status").bind(dataset, stamp, "unavailable").run();
		await recordEvent(db, dataset, /^(SOURCE_HTTP_\d{3}|SOURCE_TOO_LARGE|SOURCE_COVERAGE_TOO_LOW|INVALID_[A-Z_]+)$/.test(error.message) ? error.message : "SOURCE_FETCH_FAILED", now);
		throw error;
	}
}
async function seedJob(env, keys, now) {
	for (const key of keys) try {
		const db = env.RADAR_DB, existing = await db.prepare("SELECT market FROM radar_history WHERE market=? AND symbol=?").bind(key.market, key.symbol).first(), history = await collectSeedHistory(key.market, key.symbol, {
			now,
			recent: !!existing
		});
		await db.prepare("INSERT OR IGNORE INTO radar_history(market,symbol,available_at,payload) VALUES(?,?,?,?)").bind(key.market, key.symbol, history.availableAt, JSON.stringify(history)).run();
		const extras = key.market !== "INDEX" ? await collectSeedExtras(key.market, key.symbol, { now }) : {
			institutional: [],
			revenue: []
		}, statements = [];
		for (const [kind, rows] of [
			["prices", existing || key.market === "INDEX" ? history.bars : history.bars.slice(-1)],
			["institutional", extras.institutional],
			["revenue", extras.revenue]
		]) for (const row of rows) statements.push(factStatement(db, {
			...row,
			market: key.market,
			symbol: key.symbol
		}, kind));
		if (statements.length) await db.batch(statements);
		await db.prepare("INSERT INTO radar_seed_attempts(market,symbol,attempted_at,status) VALUES(?,?,?,'success') ON CONFLICT(market,symbol) DO UPDATE SET attempted_at=excluded.attempted_at,status=excluded.status").bind(key.market, key.symbol, new Date(now).toISOString()).run();
	} catch {
		await env.RADAR_DB.prepare("INSERT INTO radar_seed_attempts(market,symbol,attempted_at,status) VALUES(?,?,?,'failed') ON CONFLICT(market,symbol) DO UPDATE SET attempted_at=excluded.attempted_at,status=excluded.status").bind(key.market, key.symbol, new Date(now).toISOString()).run();
	}
}
async function startRun(env, session, now) {
	const correctAsOf = (/* @__PURE__ */ new Date(session + "T" + strategyConfig.schedule.initial + ":00+08:00")).toISOString(), id = session + ":" + strategyConfig.version, db = env.RADAR_DB;
	if (Date.parse(correctAsOf) > now) throw Error("FUTURE_RUN");
	const members = await currentMembers(db, correctAsOf);
	if (!(await db.prepare("INSERT OR IGNORE INTO radar_runs(id,session,as_of,config_version,expected,phase,created_at) VALUES(?,?,?,?,?,'features',?)").bind(id, session, correctAsOf, strategyConfig.version, members.length, new Date(now).toISOString()).run()).meta.changes) return;
	const benchmark = await benchmarkInput(db, correctAsOf), quotes = await latestSnapshot(db, "prices:TWSE", correctAsOf), breadth = quotes?.rows.length ? quotes.rows.filter((row) => row.change > 0).length / quotes.rows.length : null, regime = scanMarket({
		asOf: correctAsOf,
		universe: [],
		benchmark,
		context: { breadth }
	}).regime;
	const health = await strategyHealth(db, correctAsOf);
	await db.prepare("UPDATE radar_runs SET payload=? WHERE id=?").bind(JSON.stringify({
		regime,
		health
	}), id).run();
	const jobs = [];
	for (let index = 0; index < members.length; index += 3) jobs.push({
		id: session + ":features:" + index,
		body: {
			type: "features",
			runId: id,
			keys: members.slice(index, index + 3).map((stock) => ({
				symbol: stock.symbol,
				market: stock.market
			}))
		}
	});
	await enqueueBatch(env, jobs, now);
}
async function featureJob(env, body) {
	const db = env.RADAR_DB, run = await db.prepare("SELECT * FROM radar_runs WHERE id=?").bind(body.runId).first();
	if (!run) return;
	const benchmark = await benchmarkInput(db, run.as_of), reference = technicalSnapshot(benchmark, run.as_of), stocks = await stockInputs(db, body.keys, run.as_of);
	const statements = body.keys.map((key) => {
		const stock = stocks.find((item) => item.market === key.market && item.symbol === key.symbol), snapshot = stock ? technicalSnapshot(stock.bars, run.as_of) : null, eligibility = stock ? eligibleSecurity(stock, snapshot, run.as_of, reference?.last.date) : {
			eligible: false,
			reasons: ["當時股票池資料不足"]
		};
		const payload = {
			symbol: key.symbol,
			market: key.market,
			eligible: eligibility.eligible,
			reasons: eligibility.reasons,
			returns: snapshot?.returns || {}
		};
		return db.prepare("INSERT OR IGNORE INTO radar_features(run_id,market,symbol,eligible,payload) VALUES(?,?,?,?,?)").bind(run.id, key.market, key.symbol, eligibility.eligible ? 1 : 0, JSON.stringify(payload));
	});
	if (statements.length) await db.batch(statements);
}
async function scoreJob(env, body) {
	const db = env.RADAR_DB, run = await db.prepare("SELECT * FROM radar_runs WHERE id=?").bind(body.runId).first();
	if (!run) return;
	const population = (await dbRows(db.prepare("SELECT payload FROM radar_features WHERE run_id=? AND eligible=1").bind(run.id).all())).map(parse), stocks = await stockInputs(db, body.keys, run.as_of), benchmark = await benchmarkInput(db, run.as_of);
	const newsEvents = ((await latestSnapshot(db, "news", run.as_of))?.rows || []).map((event) => ({
		...event,
		sourceUrl: event.sources[0]?.url,
		companies: stocks.filter((stock) => companyMentions(event, stock).length).map((stock) => ({
			symbol: stock.symbol,
			market: stock.market,
			direct: true,
			direction: event.companyDirections?.find((item) => item.symbol === stock.symbol && item.market === stock.market)?.direction || "uncertain"
		}))
	}));
	const rowContext = await latestSnapshot(db, "prices:TWSE", run.as_of), advances = rowContext?.rows.filter((row) => Number.isFinite(row.change) && row.change > 0).length, breadth = rowContext?.rows.length ? advances / rowContext.rows.length : null;
	const result = scanMarket({
		asOf: run.as_of,
		universe: stocks,
		benchmark,
		rankUniverse: population,
		news: newsEvents,
		context: { breadth },
		strategyHealth: run.payload ? parse(run).health || {} : {},
		coverage: { complete: false }
	});
	const candidates = new Map(result.candidates.map((stock) => [stock.market + ":" + stock.symbol, stock]));
	await db.batch(body.keys.map((key) => db.prepare("INSERT OR IGNORE INTO radar_candidates(run_id,market,symbol,payload) VALUES(?,?,?,?)").bind(run.id, key.market, key.symbol, JSON.stringify(candidates.get(key.market + ":" + key.symbol) || {
		...key,
		state: "SKIP",
		score: 0,
		gaps: ["評分資料不足"]
	}))));
}
async function publishRun(env, session, now) {
	const db = env.RADAR_DB, run = await db.prepare("SELECT * FROM radar_runs WHERE session=? ORDER BY created_at DESC LIMIT 1").bind(session).first();
	if (!run) return;
	const eligible = await db.prepare("SELECT COUNT(*) AS count FROM radar_features WHERE run_id=? AND eligible=1").bind(run.id).first(), processed = await db.prepare("SELECT COUNT(*) AS count FROM radar_features WHERE run_id=?").bind(run.id).first(), all = (await dbRows(db.prepare("SELECT payload FROM radar_candidates WHERE run_id=?").bind(run.id).all())).map(parse).sort((left, right) => right.score - left.score || left.symbol.localeCompare(right.symbol)), late = now >= Date.parse(session + "T09:00:00+08:00"), complete = run.phase === "complete" && run.expected > 0 && !late;
	const benchmark = await benchmarkInput(db, run.as_of), regime = run.payload ? parse(run).regime : scanMarket({
		asOf: run.as_of,
		universe: [],
		benchmark
	}).regime;
	const watch = complete ? all.filter((stock) => stock.state === "WATCH").slice(0, 20) : [], snapshot = {
		schema: 1,
		mode: "RESEARCH",
		session,
		runId: run.id,
		asOf: run.as_of,
		publishedAt: new Date(now).toISOString(),
		configVersion: strategyConfig.version,
		status: complete ? "PUBLISHED" : late ? "LATE" : "INCOMPLETE",
		universeSize: run.expected,
		eligibleCount: eligible.count,
		processedCount: processed.count,
		high: [],
		watch,
		regime,
		validationStatus: "ACCUMULATING",
		preOpenVerified: false,
		gaps: [
			"尚無有效樣本外歷史驗證",
			"未取得盤前試撮資料",
			"隔夜產業資料尚未接齊"
		]
	};
	const paper = watch.filter((stock) => stock.primarySetup && stock.score >= regime.threshold).slice(0, 5).map((stock) => ({
		...stock,
		id: session + ":" + stock.market + ":" + stock.symbol + ":" + strategyConfig.version,
		asOf: run.as_of,
		session,
		paperOnly: true
	}));
	await savePublication(db, snapshot, paper);
}
async function outcomeJob(env, keys, now) {
	const db = env.RADAR_DB, asOf = new Date(now).toISOString();
	for (const key of keys) {
		const row = await db.prepare("SELECT payload FROM radar_signals WHERE id=?").bind(key.id).first();
		if (!row) continue;
		const signal = parse(row), stocks = await stockInputs(db, [signal], asOf);
		if (!stocks[0]) continue;
		const statements = [];
		for (const horizon of [
			5,
			10,
			20
		]) for (const exit of [
			"fixed",
			"atr",
			"trailing",
			"ma20",
			"target"
		]) {
			const result = executePaperTrade(signal, stocks[0].bars, {
				horizon,
				exit,
				evaluationAt: asOf,
				costs: TradingCostConfig
			});
			if (result.status !== "PENDING") statements.push(db.prepare("INSERT OR IGNORE INTO radar_outcomes(signal_id,horizon,exit_model,available_at,payload) VALUES(?,?,?,?,?)").bind(signal.id, horizon, exit, asOf, JSON.stringify({
				...result,
				symbol: signal.symbol,
				market: signal.market,
				strategy: signal.primarySetup,
				horizon,
				exitModel: exit
			})));
		}
		if (statements.length) await db.batch(statements);
	}
}
async function runRadarJob(env, body, now = Date.now()) {
	if (!body || typeof body.type !== "string") throw Error("INVALID_JOB");
	if (body.type === "collect") return sourceJob(env, body.dataset, now);
	if (body.type === "seed") return seedJob(env, body.keys || [], now);
	if (body.type === "start") return startRun(env, body.session, now);
	if (body.type === "features") return featureJob(env, body);
	if (body.type === "score") return scoreJob(env, body);
	if (body.type === "publish") return publishRun(env, body.session, now);
	if (body.type === "outcomes") return outcomeJob(env, body.keys || [], now);
	throw Error("UNKNOWN_JOB");
}

//#endregion
//#region scripts/radar-nightly.mjs
async function restoreResearchHistory(db, submit = submitResearch) {
	const state = await db.prepare("SELECT value FROM local_workflow_state WHERE name='history_import'").first();
	if (state?.value === "done") return {
		restored: 0,
		complete: true
	};
	let cursor = state?.value || "", restored = 0;
	for (let page = 0; page < 100; page++) {
		const result = await submit({ cursor }, { resource: "history" });
		if (!Array.isArray(result.rows) || result.rows.length > 5) throw Error("INVALID_HISTORY_CHECKPOINT");
		for (const row of result.rows) {
			if (!(["TWSE", "TPEx"].includes(row.market) && /^[1-9]\d{3}$/.test(row.symbol) || row.market === "INDEX" && row.symbol === "TAIEX") || !Number.isFinite(Date.parse(row.available_at)) || Date.parse(row.available_at) > Date.now()) throw Error("INVALID_HISTORY_CHECKPOINT");
			const history = JSON.parse(row.payload);
			if (!Array.isArray(history.bars) || !history.bars.length || history.bars.length > 400 || history.bars.some((bar) => !validBar(bar))) throw Error("INVALID_HISTORY_CHECKPOINT");
			await db.prepare("INSERT OR IGNORE INTO radar_history(market,symbol,available_at,payload) VALUES(?,?,?,?)").bind(row.market, row.symbol, row.available_at, JSON.stringify({
				market: row.market,
				symbol: row.symbol,
				availableAt: row.available_at,
				bars: history.bars,
				provenance: history.provenance
			})).run();
			restored++;
		}
		const next = result.nextCursor;
		if (next !== null && (typeof next !== "string" || next === cursor)) throw Error("INVALID_HISTORY_CHECKPOINT");
		await db.prepare("INSERT INTO local_workflow_state(name,value) VALUES('history_import',?) ON CONFLICT(name) DO UPDATE SET value=excluded.value").bind(next || "done").run();
		if (!next) return {
			restored,
			complete: true
		};
		cursor = next;
	}
	return {
		restored,
		complete: false
	};
}
async function syncPublishedSignals(db, fetcher = fetch) {
	const response = await fetcher("https://holdings-radar-research-isolated.b28228223.workers.dev/api/radar/ledger", {
		redirect: "error",
		signal: AbortSignal.timeout(2e4)
	});
	if (!response.ok) throw Error("PUBLISHED_SIGNALS_UNAVAILABLE");
	const value = await response.json();
	if (value.schema !== 1 || !Array.isArray(value.signals) || value.signals.length > 100) throw Error("INVALID_SIGNALS");
	for (const signal of value.signals) {
		if (!["TWSE", "TPEx"].includes(signal.market) || !/^[1-9]\d{3}$/.test(signal.symbol) || signal.id !== signal.session + ":" + signal.market + ":" + signal.symbol + ":" + strategyConfig.version) throw Error("INVALID_SIGNALS");
		const { results, ...payload } = signal;
		await db.prepare("INSERT OR IGNORE INTO radar_signals(id,session,market,symbol,strategy,score,reference_price,config_version,available_at,payload) VALUES(?,?,?,?,?,?,?,?,?,?)").bind(signal.id, signal.session, signal.market, signal.symbol, signal.primarySetup, signal.score, signal.referencePrice, strategyConfig.version, signal.session + "T06:00:00+08:00", JSON.stringify(payload)).run();
		for (const item of results || []) {
			const result = item.result;
			if (result?.status !== "COMPLETE") continue;
			await db.prepare("INSERT OR IGNORE INTO radar_outcomes(signal_id,horizon,exit_model,available_at,payload) VALUES(?,?,?,?,?)").bind(signal.id, item.horizon, "atr", result.availableAt, JSON.stringify(result)).run();
		}
	}
}
async function evaluateOutcomes(db, now = Date.now()) {
	const asOf = new Date(now).toISOString(), signals = await dbRows(db.prepare("SELECT id,payload FROM radar_signals ORDER BY session DESC LIMIT 100").all()), results = [];
	for (const row of signals) {
		const signal = JSON.parse(row.payload), stocks = await stockInputs(db, [signal], asOf);
		if (!stocks[0]) continue;
		for (const horizon of [
			5,
			10,
			20
		]) for (const exit of [
			"fixed",
			"atr",
			"trailing",
			"ma20",
			"target"
		]) {
			if (await db.prepare("SELECT signal_id FROM radar_outcomes WHERE signal_id=? AND horizon=? AND exit_model=?").bind(row.id, horizon, exit).first()) continue;
			const result = executePaperTrade(signal, stocks[0].bars, {
				horizon,
				exit,
				evaluationAt: asOf,
				costs: TradingCostConfig
			});
			if (result.status === "COMPLETE") results.push({
				...result,
				signalId: row.id
			});
		}
	}
	return results;
}
async function analyzeResearch(db, now = Date.now()) {
	const session = taipeiDay(now), asOf = new Date(now).toISOString();
	const members = await currentMembers(db, asOf), stocks = await stockInputs(db, members, asOf), benchmark = await benchmarkInput(db, asOf);
	if (!["TWSE", "TPEx"].every((market) => members.some((stock) => stock.market === market)) || benchmark.length < strategyConfig.minimumBars) throw Error("RESEARCH_INPUT_INCOMPLETE");
	const news = await latestSnapshot(db, "news", asOf), prices = await latestSnapshot(db, "prices:TWSE", asOf);
	const events = (news?.rows || []).map((event) => ({
		...event,
		sourceUrl: event.sources?.[0]?.url,
		companies: stocks.filter((stock) => companyMentions(event, stock).length).map((stock) => ({
			market: stock.market,
			symbol: stock.symbol,
			direct: true,
			direction: event.companyDirections?.find((item) => item.market === stock.market && item.symbol === stock.symbol)?.direction || "uncertain"
		}))
	}));
	const breadth = prices?.rows.length ? prices.rows.filter((row) => row.change > 0).length / prices.rows.length : null;
	const result = scanMarket({
		asOf,
		universe: stocks,
		benchmark,
		news: events,
		context: { breadth },
		strategyHealth: await strategyHealth(db, asOf),
		coverage: { complete: false }
	});
	const sources = await dbRows(db.prepare("SELECT * FROM radar_source_checks ORDER BY dataset").all());
	const history = await db.prepare("SELECT COUNT(*) AS count FROM radar_history h JOIN radar_members m ON m.market=h.market AND m.symbol=h.symbol WHERE m.active=1").first();
	return {
		schema: 1,
		mode: "RESEARCH",
		session,
		runId: session + ":" + strategyConfig.version,
		asOf,
		preparedAt: new Date(now).toISOString(),
		configVersion: strategyConfig.version,
		status: "DIAGNOSTIC",
		universeSize: members.length,
		processedCount: members.length,
		eligibleCount: result.eligibleCount,
		high: [],
		watch: result.watch,
		regime: result.regime,
		validationStatus: "ACCUMULATING",
		preOpenVerified: false,
		gaps: [
			"尚無有效樣本外歷史驗證",
			"未取得盤前試撮資料",
			"隔夜產業資料尚未接齊",
			...history.count < members.length ? ["部分股票歷史回補尚未完成，資料不足的股票不列入觀察。"] : [],
			...sources.filter((source) => source.status !== "success").map((source) => "來源暫時不可用：" + source.dataset)
		],
		collection: {
			universeSize: members.length,
			historyReady: history.count,
			sources
		}
	};
}
async function buildResearch(db, now = Date.now()) {
	return validatePrepared({
		...await analyzeResearch(db, now),
		status: "READY"
	}, now);
}
async function prepareInputs(db, { now = Date.now(), seedLimit = 120, onProgress = () => {} } = {}) {
	const env = { RADAR_DB: db };
	for (const dataset of Object.keys(radarSources)) try {
		await runRadarJob(env, {
			type: "collect",
			dataset
		}, now);
		onProgress({
			phase: "collect",
			dataset,
			status: "success"
		});
	} catch {
		onProgress({
			phase: "collect",
			dataset,
			status: "unavailable"
		});
	}
	await runRadarJob(env, {
		type: "seed",
		keys: [{
			market: "INDEX",
			symbol: "TAIEX"
		}]
	}, now);
	const turnover = /* @__PURE__ */ new Map();
	for (const market of ["TWSE", "TPEx"]) for (const row of (await latestSnapshot(db, "prices:" + market, new Date(now).toISOString()))?.rows || []) turnover.set(market + ":" + row.symbol, row.turnover || 0);
	const keys = (await dbRows(db.prepare("SELECT m.market,m.symbol,(h.symbol IS NULL) AS missing,COALESCE(a.attempted_at,'') AS attempted FROM radar_members m LEFT JOIN radar_history h ON h.market=m.market AND h.symbol=m.symbol LEFT JOIN radar_seed_attempts a ON a.market=m.market AND a.symbol=m.symbol WHERE m.active=1 AND (h.symbol IS NULL OR m.market='TPEx') AND (a.attempted_at IS NULL OR a.attempted_at<?)").bind((/* @__PURE__ */ new Date(now - 216e5)).toISOString()).all())).sort((left, right) => right.missing - left.missing || left.attempted.localeCompare(right.attempted) || (turnover.get(right.market + ":" + right.symbol) || 0) - (turnover.get(left.market + ":" + left.symbol) || 0) || left.symbol.localeCompare(right.symbol)).slice(0, Math.max(0, Math.min(seedLimit, 120)));
	for (const key of keys) await runRadarJob(env, {
		type: "seed",
		keys: [key]
	}, Date.now());
	return {
		attempted: keys.length,
		sources: await dbRows(db.prepare("SELECT dataset,status,row_count FROM radar_source_checks ORDER BY dataset").all())
	};
}
async function publisherToken(endpoint, env = process.env, fetcher = fetch) {
	if (!env.ACTIONS_ID_TOKEN_REQUEST_URL || !env.ACTIONS_ID_TOKEN_REQUEST_TOKEN) throw Error("PUBLISH_AUTH_REQUIRED");
	const url = new URL(env.ACTIONS_ID_TOKEN_REQUEST_URL);
	if (url.protocol !== "https:" || !url.hostname.endsWith(".actions.githubusercontent.com")) throw Error("INVALID_OIDC_ENDPOINT");
	url.searchParams.set("audience", endpoint.origin);
	const response = await fetcher(url, {
		headers: { Authorization: "Bearer " + env.ACTIONS_ID_TOKEN_REQUEST_TOKEN },
		redirect: "error",
		signal: AbortSignal.timeout(15e3)
	});
	if (!response.ok) throw Error("OIDC_FAILED");
	const value = await response.json();
	if (typeof value.value !== "string") throw Error("OIDC_FAILED");
	return value.value;
}
async function submitResearch(value, { check = false, resource = "prepare", env = process.env, fetcher = fetch } = {}) {
	const endpoint = new URL(env.RADAR_WORKER_URL || "https://holdings-radar-research-isolated.b28228223.workers.dev");
	if (endpoint.origin !== "https://holdings-radar-research-isolated.b28228223.workers.dev") throw Error("INVALID_PUBLISH_TARGET");
	const token = await publisherToken(endpoint, env, fetcher);
	if (![
		"prepare",
		"outcomes",
		"history"
	].includes(resource)) throw Error("INVALID_PUBLISH_RESOURCE");
	const response = await fetcher(new URL("/api/radar/research/" + (check ? "check" : resource), endpoint), {
		method: "POST",
		headers: {
			Authorization: "Bearer " + token,
			"Content-Type": "application/json"
		},
		body: check ? "{}" : JSON.stringify(value),
		redirect: "error",
		signal: AbortSignal.timeout(3e4)
	});
	if (!response.ok) throw Error("PUBLISH_HTTP_" + response.status);
	const result = await response.json();
	if (result.ok !== true) throw Error("PUBLISH_REJECTED");
	return result;
}
async function main() {
	const phase = process.env.RADAR_PHASE || "auto", now = Date.now(), day = taipeiDay(now), time = new Intl.DateTimeFormat("en-GB", {
		timeZone: "Asia/Taipei",
		hour: "2-digit",
		minute: "2-digit",
		hour12: false
	}).format(now);
	if (![
		"auto",
		"prepare",
		"research",
		"check"
	].includes(phase)) throw Error("INVALID_PHASE");
	if (phase === "check") {
		console.log(JSON.stringify(await submitResearch(null, { check: true })));
		return;
	}
	const compute = phase === "research" || phase === "auto" && time >= "03:20" && time < "05:40";
	if (compute && ([0, 6].includes((/* @__PURE__ */ new Date(day + "T00:00:00Z")).getUTCDay()) || time >= "05:40")) {
		console.log(JSON.stringify({
			skipped: true,
			reason: "OUTSIDE_RESEARCH_WINDOW"
		}));
		return;
	}
	const filename = resolve(process.env.RADAR_CACHE_FILE || "radar-runtime/checkpoint.sqlite"), db = openResearchDb(filename);
	const originalFetch = globalThis.fetch;
	let calls = 0, blocked = false;
	globalThis.fetch = async (input, init = {}) => {
		if (new URL(typeof input === "string" ? input : input.url || input).hostname === "api.finmindtrade.com") {
			if (blocked || ++calls > 400) throw Error("SOURCE_REQUEST_BUDGET");
			if (process.env.FINMIND_TOKEN) {
				const headers = new Headers(init.headers);
				headers.set("Authorization", "Bearer " + process.env.FINMIND_TOKEN);
				init = {
					...init,
					headers
				};
			}
			const response = await originalFetch(input, init);
			if (response.status === 429 || response.status === 402) blocked = true;
			return response;
		}
		return originalFetch(input, init);
	};
	try {
		console.log(JSON.stringify({
			phase: "history_checkpoint",
			...await restoreResearchHistory(db)
		}));
		try {
			await syncPublishedSignals(db);
		} catch {
			console.log(JSON.stringify({
				phase: "paper_history",
				status: "unavailable",
				cachedHistoryRetained: true
			}));
		}
		const collection = await prepareInputs(db, {
			seedLimit: compute ? 0 : 120,
			onProgress: (value) => console.log(JSON.stringify(value))
		});
		const outcomes = await evaluateOutcomes(db);
		for (let index = 0; index < outcomes.length; index += 20) {
			const group = outcomes.slice(index, index + 20);
			try {
				await submitResearch(group, { resource: "outcomes" });
			} catch {
				console.log(JSON.stringify({
					phase: "paper_outcomes",
					status: "upload_pending"
				}));
				break;
			}
			for (const row of group) await db.prepare("INSERT OR IGNORE INTO radar_outcomes(signal_id,horizon,exit_model,available_at,payload) VALUES(?,?,?,?,?)").bind(row.signalId, row.horizon, row.exitModel, row.availableAt, JSON.stringify(row)).run();
		}
		if (!compute) {
			console.log(JSON.stringify({
				phase: "prepare",
				...collection
			}));
			return;
		}
		if (Date.now() >= Date.parse(day + "T" + "05:40" + ":00+08:00")) throw Error("RESEARCH_DEADLINE_PASSED");
		const prepared = await buildResearch(db, Date.now()), output = resolve("radar-runtime/prepared.json");
		await mkdir(dirname(output), { recursive: true });
		await writeFile(output, JSON.stringify(prepared));
		const result = await submitResearch(prepared);
		console.log(JSON.stringify({
			phase: "ready",
			session: prepared.session,
			processedCount: prepared.processedCount,
			eligibleCount: prepared.eligibleCount,
			watchCount: prepared.watch.length,
			...result
		}));
	} finally {
		globalThis.fetch = originalFetch;
		db.close();
	}
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) main().catch((error) => {
	console.error(JSON.stringify({
		phase: "research",
		error: /^(RESEARCH_[A-Z_]+|PUBLISH_[A-Z_0-9]+|OIDC_FAILED|INVALID_[A-Z_]+)$/.test(error.message) ? error.message : "RESEARCH_FAILED",
		previousPublicationRetained: true
	}));
	process.exitCode = 1;
});

//#endregion
export { analyzeResearch, buildResearch, evaluateOutcomes, prepareInputs, publisherToken, restoreResearchHistory, submitResearch, syncPublishedSignals };