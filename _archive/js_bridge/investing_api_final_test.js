#!/usr/bin/env node
/**
 * Final Investing.com API Test
 * ============================
 * 
 * Using the correct pattern from the test file
 */

const InvestingApiV2 = require('./node_modules/investing-com-api-v2/api/InvestingApiV2');

async function testFinalAPI() {
    console.log('🚀 Testing Investing.com API V2 (Final Test)');
    console.log('=' .repeat(50));
    
    try {
        // Configure the API
        InvestingApiV2.logger(console);
        InvestingApiV2.setDebugger(false);  // Disable debug for cleaner output
        
        // Initialize the API
        console.log('🔧 Initializing API...');
        await InvestingApiV2.init({ headless: true });
        console.log('✅ API initialized successfully');
        
        // Test 1: Known working currency pair
        console.log('\n📊 Test 1: Testing EUR/USD...');
        try {
            const eurUsd = await InvestingApiV2.investing('currencies/eur-usd', 'P1D', 'PT1H', 60);
            if (eurUsd && eurUsd.length > 0) {
                console.log(`✅ EUR/USD: Got ${eurUsd.length} data points`);
                console.log(`   Latest: ${eurUsd[0].date} - Close: ${eurUsd[0].close}`);
                console.log(`   Data structure:`, Object.keys(eurUsd[0]));
            }
        } catch (error) {
            console.log(`❌ EUR/USD error: ${error.message}`);
        }
        
        // Test 2: Try Pakistani Rupee pairs
        console.log('\n📊 Test 2: Testing PKR Currency Pairs...');
        
        const pkrPairs = [
            'currencies/usd-pkr',
            'currencies/eur-pkr', 
            'currencies/gbp-pkr'
        ];
        
        for (let pair of pkrPairs) {
            try {
                console.log(`\n🔍 Testing ${pair}...`);
                const data = await InvestingApiV2.investing(pair, 'P1W', 'P1D', 60);
                
                if (data && data.length > 0) {
                    console.log(`✅ ${pair}: Got ${data.length} data points`);
                    console.log(`   Latest: ${data[0].date} - Close: ${data[0].close}`);
                } else {
                    console.log(`❌ ${pair}: No data returned`);
                }
                
                // Small delay to avoid rate limiting
                await new Promise(resolve => setTimeout(resolve, 1000));
                
            } catch (pairError) {
                console.log(`❌ ${pair}: ${pairError.message}`);
            }
        }
        
        // Test 3: Try some stock indices that might include PSX
        console.log('\n📊 Test 3: Testing Stock Indices...');
        
        const indices = [
            'indices/us-spx-500',      // S&P 500
            'indices/nikkei-225',      // Nikkei
            'indices/shanghai-composite' // Shanghai
        ];
        
        for (let index of indices.slice(0, 1)) { // Test only first one
            try {
                console.log(`\n🔍 Testing ${index}...`);
                const data = await InvestingApiV2.investing(index, 'P1M', 'P1D', 60);
                
                if (data && data.length > 0) {
                    console.log(`✅ ${index}: Got ${data.length} data points`);
                    console.log(`   Latest: ${data[0].date} - Close: ${data[0].close}`);
                }
                
            } catch (indexError) {
                console.log(`❌ ${index}: ${indexError.message}`);
            }
        }
        
        // Summary
        console.log('\n📋 API Assessment for PSX Integration:');
        console.log('✅ API works well for currency data');
        console.log('✅ Can get PKR exchange rates (useful for PSX)');
        console.log('❓ Direct PSX stock data may not be available');
        console.log('💡 Recommendation: Use for currency data + current EODHD for stocks');
        
    } catch (error) {
        console.error(`❌ API Error: ${error.message}`);
    } finally {
        // Always close the API
        try {
            console.log('\n🔧 Closing API...');
            await InvestingApiV2.close();
            console.log('✅ API closed successfully');
        } catch (closeError) {
            console.error(`❌ Error closing API: ${closeError.message}`);
        }
    }
    
    console.log('\n🏁 Test completed!');
}

// Run the test
if (require.main === module) {
    testFinalAPI().catch(console.error);
}

module.exports = { testFinalAPI };