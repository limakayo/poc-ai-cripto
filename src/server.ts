import 'dotenv/config';
import { ChatOpenAI } from "@langchain/openai";
import { AgentExecutor, createOpenAIFunctionsAgent } from "langchain/agents";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { BufferMemory, BaseMemory } from "langchain/memory";
import { 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate, 
    ChatPromptTemplate,
    AIMessagePromptTemplate 
} from "@langchain/core/prompts";
// import { TwitterApi } from 'twitter-api-v2';
import { z } from "zod";

// Interfaces
interface BinanceTickerResponse {
    symbol: string;
    lastPrice: string;
    priceChangePercent: string;
    volume: string;
    quoteVolume: string;
    highPrice: string;
    lowPrice: string;
}

interface FearGreedResponse {
    data: {
        value: string;
        value_classification: string;
        timestamp: string;
    }[];
}

interface HistoricalPriceResponse {
    Data: {
        close: number;
        volumeto: number;
        volumefrom: number;
        time: number;
    }[];
}

interface PredictionAnalysis {
    targetRange: string;
    targetDate: string;
    currentPrice: string;
    confidence: string;
    supportingFactors: string[];
    technicalIndicators: {
        fearGreedIndex: string;
        fearGreedValue: string;
        trend: string;
    };
    keyRisks: string[];
}

// Twitter Client Configuration
// const twitterClient = new TwitterApi({
//     appKey: process.env.TWITTER_APP_KEY || '',
//     appSecret: process.env.TWITTER_APP_SECRET || '',
//     accessToken: process.env.TWITTER_ACCESS_TOKEN || '',
//     accessSecret: process.env.TWITTER_ACCESS_SECRET || '',
// });

// Tools
const btcDataTool = new DynamicStructuredTool({
    name: "get_btc_data",
    description: "Get current Bitcoin price data from Binance",
    schema: z.object({
        symbol: z.string().describe("Trading pair symbol (always BTCUSDT)")
    }),
    async func({ symbol }): Promise<string> {
        try {
            const url = `https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT`;
            const response = await fetch(url);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            
            const data: BinanceTickerResponse = await response.json();
            
            const formattedData = {
                lastPrice: parseFloat(data.lastPrice).toFixed(2),
                priceChangePercent: parseFloat(data.priceChangePercent).toFixed(2),
                volume: parseFloat(data.volume).toFixed(2),
                quoteVolume: parseFloat(data.quoteVolume).toFixed(2),
                highPrice: parseFloat(data.highPrice).toFixed(2),
                lowPrice: parseFloat(data.lowPrice).toFixed(2)
            };

            console.log("BINANCE:", formattedData);
            return JSON.stringify(formattedData, null, 2);
        } catch (error) {
            console.error("Erro ao buscar dados do Bitcoin:", error);
            throw error;
        }
    }
});

const fearGreedTool = new DynamicStructuredTool({
    name: "get_fear_greed_index",
    description: "Get Bitcoin Fear & Greed Index data",
    schema: z.object({}),
    async func(): Promise<string> {
        try {
            const response = await fetch('https://api.alternative.me/fng/?limit=30');
            const data: FearGreedResponse = await response.json();
            console.log("Fear & Greed Index:", data);
            return JSON.stringify(data, null, 2);
        } catch (error) {
            console.error("Error fetching Fear & Greed Index:", error);
            throw error;
        }
    }
});

const historicalDataTool = new DynamicStructuredTool({
    name: "get_historical_prices",
    description: "Get Bitcoin historical price data",
    schema: z.object({
        days: z.number().describe("Number of days of historical data to fetch")
    }),
    async func({ days }): Promise<string> {
        try {
            const response = await fetch(
                `https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&limit=${days}`
            );
            const data: HistoricalPriceResponse = await response.json();
            console.log("Historical Data:", data);
            return JSON.stringify(data, null, 2);
        } catch (error) {
            console.error("Error fetching historical data:", error);
            throw error;
        }
    }
});

// Agents
async function createRealtimeAgent(memory: BaseMemory): Promise<AgentExecutor> {
    const model = new ChatOpenAI({
        temperature: 0,
        modelName: "gpt-3.5-turbo",
        openAIApiKey: process.env.OPENAI_API_KEY,
        maxTokens: 1000
    });

    const systemMessage = `Você é um agente especializado em reportar dados do Bitcoin.

FORMATE OS DADOS EXATAMENTE ASSIM:

**Bitcoin (BTCUSDT)**
- Preço Atual: $[lastPrice]
- Variação 24h: [priceChangePercent]%
- Volume: [volume] BTC / $[quoteVolume] USDT
- Alta: $[highPrice]
- Baixa: $[lowPrice]

Use os números EXATAMENTE como fornecidos.`;

    const chatPrompt = ChatPromptTemplate.fromMessages([
        SystemMessagePromptTemplate.fromTemplate(systemMessage),
        new MessagesPlaceholder("chat_history"),
        HumanMessagePromptTemplate.fromTemplate("{input}"),
        AIMessagePromptTemplate.fromTemplate("{agent_scratchpad}")
    ]);

    const agent = await createOpenAIFunctionsAgent({
        llm: model,
        tools: [btcDataTool],
        prompt: chatPrompt
    });

    return new AgentExecutor({
        agent,
        tools: [btcDataTool],
        memory,
        verbose: false
    });
}

async function createPredictionAgent(memory: BaseMemory): Promise<AgentExecutor> {
    const model = new ChatOpenAI({
        temperature: 0.7,
        modelName: "gpt-3.5-turbo",
        openAIApiKey: process.env.OPENAI_API_KEY,
        maxTokens: 1000
    });

    const systemMessage = `Você é um analista de previsão do Bitcoin.
Use os dados disponíveis para avaliar se o Bitcoin atingirá determinados preços.

ANÁLISE OBRIGATÓRIA:
1. Dados históricos (use get_historical_prices)
2. Índice Fear & Greed (use get_fear_greed_index)
3. Dados atuais (já fornecidos)

FORMATO DA RESPOSTA:

ANÁLISE PREDITIVA BITCOIN
-------------------------
Alvo: [Faixa de preço e data]
Confiança: [Baixa/Média/Alta]

FATORES POSITIVOS:
1. [fator concreto baseado em dados]
2. [fator concreto baseado em dados]
3. [fator concreto baseado em dados]

FATORES DE RISCO:
1. [risco baseado em dados]
2. [risco baseado em dados]
3. [risco baseado em dados]

INDICADORES TÉCNICOS:
- Fear & Greed: [valor atual] ([classificação])
- Tendência: [tendência baseada nos dados históricos]
- Momentum: [análise do momentum atual]

CONCLUSÃO:
[Análise objetiva baseada apenas nos dados apresentados]`;

    const chatPrompt = ChatPromptTemplate.fromMessages([
        SystemMessagePromptTemplate.fromTemplate(systemMessage),
        new MessagesPlaceholder("chat_history"),
        HumanMessagePromptTemplate.fromTemplate("{input}"),
        AIMessagePromptTemplate.fromTemplate("{agent_scratchpad}")
    ]);

    const tools = [fearGreedTool, historicalDataTool];

    const agent = await createOpenAIFunctionsAgent({
        llm: model,
        tools,
        prompt: chatPrompt
    });

    return new AgentExecutor({
        agent,
        tools,
        memory,
        verbose: false
    });
}

// Twitter Functions
async function createPredictionTweets(analysis: PredictionAnalysis): Promise<string[]> {
    const tweets = [
        // Market Data Tweet
        `#Bitcoin Market Data 📊\n` +
        `Current Price: $${analysis.currentPrice}\n` +
        `Target Analysis: ${analysis.targetRange} by ${analysis.targetDate}\n` +
        `Confidence Level: ${analysis.confidence}\n` +
        `#BTC #Crypto`,

        // Technical Analysis Tweet
        `#BTC Technical Data 📈\n` +
        `Fear & Greed: ${analysis.technicalIndicators.fearGreedValue} (${analysis.technicalIndicators.fearGreedIndex})\n` +
        `Market Trend: ${analysis.technicalIndicators.trend}\n` +
        `#Bitcoin #Trading`,

        // Supporting Factors Tweet
        `Supporting Data for #Bitcoin:\n` +
        analysis.supportingFactors.slice(0, 2).map(factor => `• ${factor}`).join('\n') +
        `\n#BTC #CryptoAnalysis`,

        // Risk Assessment Tweet
        `#BTC Risk Factors:\n` +
        analysis.keyRisks.slice(0, 2).map(risk => `• ${risk}`).join('\n') +
        `\n#Bitcoin #CryptoMarkets`
    ];

    return tweets;
}

// async function postPredictionThread(tweets: string[]): Promise<void> {
//     try {
//         let lastTweetId: string | undefined;

//         for (const tweet of tweets) {
//             const { data: createdTweet } = await twitterClient.v2.tweet(
//                 tweet,
//                 lastTweetId ? { reply: { in_reply_to_tweet_id: lastTweetId } } : undefined
//             );

//             lastTweetId = createdTweet.id;
//             console.log(`Tweet postado: ${tweet}`);
//             await new Promise(resolve => setTimeout(resolve, 2000));
//         }

//         console.log('Thread de previsão postada com sucesso!');
//     } catch (error) {
//         console.error('Erro ao postar thread:', error);
//         throw error;
//     }
// }

// Data Extraction Functions
async function analyzeBitcoinAndTweet(targetPrice: string, targetDate: string): Promise<void> {
    try {
        const memory = new BufferMemory({
            returnMessages: true,
            memoryKey: "chat_history",
            outputKey: "output"
        }) as BaseMemory;

        // Get Real-time Data
        console.log("\n=== Dados em Tempo Real ===");
        const realtimeAgent = await createRealtimeAgent(memory);
        const realtimeData = await realtimeAgent.invoke({
            input: "Get current Bitcoin data"
        });
        
        // Parse realtime data using regex
        const output = realtimeData.output;
        const parsedRealtimeData = {
            lastPrice: output.match(/Preço Atual: \$([0-9.]+)/)?.[1] || "",
            priceChangePercent: output.match(/Variação 24h: ([0-9.-]+)%/)?.[1] || "",
            volume: output.match(/Volume: ([0-9.]+) BTC/)?.[1] || "",
            quoteVolume: output.match(/\/ \$([0-9.]+) USDT/)?.[1] || "",
            highPrice: output.match(/Alta: \$([0-9.]+)/)?.[1] || "",
            lowPrice: output.match(/Baixa: \$([0-9.]+)/)?.[1] || ""
        };

        console.log("Parsed Realtime Data:", parsedRealtimeData);

        // Get Prediction Analysis
        console.log("\n=== Análise Preditiva ===");
        const predictionAgent = await createPredictionAgent(memory);
        const predictionData = await predictionAgent.invoke({
            input: `Considerando os dados atuais acima, analise se o Bitcoin atingirá ${targetPrice} em ${targetDate}. 
                   Use os dados históricos e o índice Fear & Greed para fundamentar sua análise.`
        });
        
        // Extract and Structure Analysis
        const analysis = extractAnalysis(parsedRealtimeData, predictionData.output);
        analysis.targetRange = targetPrice;
        analysis.targetDate = targetDate;

        // Create and Post Tweets
        const tweets = await createPredictionTweets(analysis);
        
        tweets.forEach(tweet => console.log("Tweet:", tweet));
    } catch (error) {
        console.error("Erro na análise:", error);
        throw error;
    }
}

function extractAnalysis(realtimeData: any, predictionOutput: string): PredictionAnalysis {
    try {
        const currentPrice = realtimeData.lastPrice;
        console.log("Current Price for Analysis:", currentPrice);

        const confidence = predictionOutput.match(/Confiança: (.*?)\n/)?.[1] || "";
        
        const factorsMatch = predictionOutput.match(/FATORES POSITIVOS:\n((?:1\. .*\n?)*)/);
        const supportingFactors = factorsMatch 
            ? factorsMatch[1]
                .split('\n')
                .filter(line => line.trim().startsWith('1.') || line.trim().startsWith('2.'))
                .map(line => line.substring(3).trim())
            : [];

        const risksMatch = predictionOutput.match(/FATORES DE RISCO:\n((?:1\. .*\n?)*)/);
        const risks = risksMatch
            ? risksMatch[1]
                .split('\n')
                .filter(line => line.trim().startsWith('1.') || line.trim().startsWith('2.'))
                .map(line => line.substring(3).trim())
            : [];

        const fearGreedMatch = predictionOutput.match(/Fear & Greed: (.*?) \((.*?)\)/);
        const trendMatch = predictionOutput.match(/Tendência: (.*?)\n/);

        const analysis: PredictionAnalysis = {
            targetRange: "",
            targetDate: "",
            currentPrice,
            confidence,
            supportingFactors,
            technicalIndicators: {
                fearGreedValue: fearGreedMatch?.[1] || "",
                fearGreedIndex: fearGreedMatch?.[2] || "",
                trend: trendMatch?.[1] || ""
            },
            keyRisks: risks
        };

        return analysis;
    } catch (error) {
        console.error('Erro ao extrair análise:', error);
        throw error;
    }
}

// Main Function
async function main(): Promise<void> {
    await analyzeBitcoinAndTweet("$97k-$100k", "31 de dezembro de 2024");
}

// Execute
main().catch(console.error);