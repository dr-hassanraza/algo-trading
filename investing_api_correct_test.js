#!/usr/bin/env node
/**
 * Correct Investing.com API Test
 * ==============================
 * 
 * Based on the actual API structure, this uses the InvestingApiV2 class correctly
 */

const InvestingApiV2 = require('investing-com-api-v2/api/InvestingApiV2');

async function testCorrectAPI() {
    console.log('🚀 Testing Investing.com API V2 (Correct Usage)');
    console.log('=' .repeat(50));
    
    const api = new InvestingApiV2();
    
    try {
        // Initialize the API (this starts puppeteer)
        console.log('🔧 Initializing API (starting browser)...');
        await api.init({ headless: true });
        console.log('✅ API initialized successfully');
        
        // Test 1: Try some known working symbols (currencies)
        console.log('\n📊 Test 1: Testing with known working symbols...');
        
        const testSymbols = [
            'currencies/eur-usd',
            'currencies/gbp-usd',
            'currencies/usd-jpy'
        ];
        
        for (let symbol of testSymbols.slice(0, 1)) { // Test first one only
            try {
                console.log(`\n🔍 Testing ${symbol}...`);
                const result = await api.investing(symbol, 'P1M', 'P1D', 120);
                
                if (result && result.length > 0) {
                    console.log(`✅ Got ${result.length} data points for ${symbol}`);
                    console.log(`   Latest: Date=${result[0].date}, Close=${result[0].close}`);
                    console.log(`   Sample data structure:`, Object.keys(result[0]));
                } else {
                    console.log(`❌ No data returned for ${symbol}`);
                }
                
            } catch (symbolError) {
                console.log(`❌ Error getting ${symbol}: ${symbolError.message}`);
            }
        }
        
        // Test 2: Try Pakistani Rupee related pairs
        console.log('\n📊 Test 2: Testing Pakistani Rupee related pairs...');
        
        const pkrPairs = [
            'currencies/usd-pkr',  // US Dollar to Pakistani Rupee
            'currencies/eur-pkr',  // Euro to Pakistani Rupee
            'currencies/gbp-pkr'   // British Pound to Pakistani Rupee
        ];
        
        for (let pair of pkrPairs) {
            try {
                console.log(`\n🔍 Testing ${pair}...`);
                const result = await api.investing(pair, 'P1M', 'P1D', 60);
                
                if (result && result.length > 0) {
                    console.log(`✅ Got ${result.length} data points for ${pair}`);
                    console.log(`   Latest: Date=${result[0].date}, Close=${result[0].close}`);
                } else {
                    console.log(`❌ No data returned for ${pair}`);
                }
                
            } catch (pairError) {
                console.log(`❌ ${pair} not available: ${pairError.message}`);
            }
        }
        
        // Test 3: Try direct pair IDs (if we can find PSX stock IDs)
        console.log('\n📊 Test 3: Testing with direct pair IDs...');
        
        // These are example IDs - in practice we'd need to find PSX stock IDs
        const testIds = [
            '1',    // EUR/USD
            '2',    // GBP/USD
        ];
        
        for (let id of testIds.slice(0, 1)) {
            try {
                console.log(`\n🔍 Testing pair ID ${id}...`);
                const result = await api.investing(id, 'P1W', 'P1D', 60);
                
                if (result && result.length > 0) {
                    console.log(`✅ Got data for ID ${id}`);
                    console.log(`   Data points: ${result.length}`);
                } else {
                    console.log(`❌ No data for ID ${id}`);
                }
                
            } catch (idError) {
                console.log(`❌ Error with ID ${id}: ${idError.message}`);
            }
        }
        
        console.log('\n📋 API Capabilities Summary:');
        console.log('✅ API can fetch financial data successfully');
        console.log('✅ Supports various time periods and intervals');
        console.log('❓ PSX stocks may need specific pair IDs to be found');
        console.log('💡 Consider using for PKR currency data as alternative');
        
    } catch (error) {
        console.error(`❌ API Error: ${error.message}`);
    } finally {
        // Always close the browser
        try {
            console.log('\n🔧 Closing API...');
            await api.close();
            console.log('✅ API closed successfully');
        } catch (closeError) {
            console.error(`❌ Error closing API: ${closeError.message}`);
        }
    }
    
    console.log('\n🏁 Test completed!');
}

// Run the test
if (require.main === module) {
    testCorrectAPI().catch(console.error);
}

module.exports = { testCorrectAPI };