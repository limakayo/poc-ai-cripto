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

// Interface for Binance API response
interface BinanceTickerResponse {
    symbol: string;
    lastPrice: string;
    priceChangePercent: string;
    volume: string;
    quoteVolume: string;
    highPrice: string;
    lowPrice: string;
}

// Crypto Data Tool
const cryptoDataTool = new DynamicStructuredTool({
    name: "get_btc_data",
    description: "Get current Bitcoin price data from Binance",
    schema: {
        type: "object",
        properties: {
            symbol: {
                type: "string",
                description: "Trading pair symbol (always BTCUSDT)"
            }
        },
        required: ["symbol"]
    },
    async func({ }): Promise<string> {
        try {
            const url = `https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT`;
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data: BinanceTickerResponse = await response.json();
            
            // Format numbers with exactly 2 decimal places
            const formattedData = {
                lastPrice: parseFloat(data.lastPrice).toFixed(2),
                priceChangePercent: parseFloat(data.priceChangePercent).toFixed(2),
                volume: parseFloat(data.volume).toFixed(2),
                quoteVolume: parseFloat(data.quoteVolume).toFixed(2),
                highPrice: parseFloat(data.highPrice).toFixed(2),
                lowPrice: parseFloat(data.lowPrice).toFixed(2)
            };

            console.log("Formatted Data:", formattedData);
            return JSON.stringify(formattedData, null, 2);
        } catch (error) {
            console.error("Erro ao buscar dados do Bitcoin:", error);
            throw error;
        }
    }
});

async function createAnalystAgent(memory: BaseMemory): Promise<AgentExecutor> {
    const model = new ChatOpenAI({
        temperature: 0,
        modelName: "gpt-3.5-turbo",
        openAIApiKey: process.env.OPENAI_API_KEY,
    });

    const systemMessage = `Você é um agente especializado em reportar dados do Bitcoin.
Ao receber dados da API Binance via get_btc_data:

1. FORMATE EXATAMENTE ASSIM:

**Bitcoin (BTCUSDT)**
- Preço Atual: $[lastPrice]
- Variação 24h: [priceChangePercent]%
- Volume: [volume] BTC / $[quoteVolume] USDT
- Alta: $[highPrice]
- Baixa: $[lowPrice]

2. Tweets:
1. #BTC Preço: $[lastPrice] ([priceChangePercent]%) [📉 se negativo, 📈 se positivo]
2. #BTC Volume 24h: [volume] BTC ($[quoteVolume] USDT)
3. #Bitcoin 24h - Alta: $[highPrice] | Baixa: $[lowPrice]

REGRAS IMPORTANTES:
- Use os números EXATAMENTE como fornecidos
- NÃO adicione ou remova casas decimais
- NÃO adicione vírgulas
- NÃO faça nenhuma modificação nos números

Os valores já virão formatados com 2 casas decimais.`;

    const chatPrompt = ChatPromptTemplate.fromMessages([
        SystemMessagePromptTemplate.fromTemplate(systemMessage),
        new MessagesPlaceholder("chat_history"),
        HumanMessagePromptTemplate.fromTemplate("{input}"),
        AIMessagePromptTemplate.fromTemplate("{agent_scratchpad}")
    ]);

    const agent = await createOpenAIFunctionsAgent({
        llm: model,
        tools: [cryptoDataTool],
        prompt: chatPrompt
    });

    return new AgentExecutor({
        agent,
        tools: [cryptoDataTool],
        memory,
        verbose: true
    });
}

async function analyzeBitcoin(): Promise<any> {
    console.log("Iniciando análise do Bitcoin...");
    
    const memory = new BufferMemory({
        returnMessages: true,
        memoryKey: "chat_history",
        outputKey: "output"
    }) as BaseMemory;

    const analyst = await createAnalystAgent(memory);

    try {
        const analysis = await analyst.invoke({
            input: "Use get_btc_data para obter e apresentar os dados atuais do Bitcoin (BTCUSDT). Mantenha todos os números exatamente como retornados pela API."
        });

        console.log("\n=== Análise do Bitcoin ===");
        console.log(analysis.output);

        return analysis;
    } catch (error) {
        console.error("Erro na análise:", error);
        throw error;
    }
}

// Main Function
async function main(): Promise<void> {
    await analyzeBitcoin();
}

// Execute
main().catch(console.error);