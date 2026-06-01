import streamlit as st
import folium
from streamlit_folium import st_folium
import json
import os
from engine import (
    InvestmentConstraints,
    calculate_dynamic_portfolio_analysis
)

# Set page configuration with a premium dark theme styling
st.set_page_config(
    page_title="Geo-Invest AI: Location Intelligence Specialist",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS to make it look stunningly premium and dark mode styled
st.markdown("""
<style>
    body {
        color: #f4f4f5;
        background-color: #09090b;
    }
    .stDeployButton {
        display: none;
    }
    /* Style headings */
    h1, h2, h3 {
        color: #14b8a6 !important;
        font-family: 'Outfit', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

st.title("🗺️ Geo-Invest AI: Location Intelligence Specialist")
st.markdown("### Multi-Modal Agentic RAG & Metaheuristic Investment Optimizer")
st.markdown("---")

# 1. Sidebar Inputs
with st.sidebar:
    st.header("🏢 Strategy Configuration")
    
    api_key = st.text_input("🔑 Gemini API Key", type="password", help="Input your free Gemini API Key for live AI generation.")
    
    no_business_idea = st.checkbox("🤷 I don't have a business idea", value=False, help="Toggle to get 5 tailored suggestions for your location.")
    
    st.markdown("---")
    st.subheader("📍 Target Location & Concept")
    user_prompt = st.text_area(
        "Enter Prompt (Location + Business Type)",
        placeholder="e.g. provide business idea in the area perumalburam, tirunelveli",
        help="Describe what you want to start and where in India, or just enter the city if you have no business idea."
    )
    
    submit_btn = st.button("🚀 Evolve Investment Strategy", use_container_width=True)

# 2. Execution logic
if submit_btn:
    if not api_key:
        st.error("⚠️ Gemini API Key is required! Please paste your key in the sidebar.")
    elif not user_prompt:
        st.error("⚠️ Please enter a target location or concept prompt first.")
    else:
        with st.spinner("Executing Neuro-Agentic routing layers and spatial iterations..."):
            try:
                # Build Constraints
                constraints = InvestmentConstraints(
                    user_prompt=user_prompt,
                    no_business_idea=no_business_idea,
                    gemini_api_key=api_key
                )
                
                # Run dynamic analysis
                payload = calculate_dynamic_portfolio_analysis(constraints)
                
                # Split display canvas
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.markdown("#### 📍 Optimized Geographic Layout Map")
                    
                    sites = payload.recommended_locations
                    if sites:
                        # Center map around first node
                        base_lat = sites[0].latitude
                        base_lon = sites[0].longitude
                        
                        # Leaflet map matching CartoDB Dark Matter
                        m = folium.Map(location=[base_lat, base_lon], zoom_start=13, tiles="CartoDB dark_matter")
                        
                        # Add Marker Nodes
                        lat_lons = []
                        for idx, site in enumerate(sites):
                            lat, lon = site.latitude, site.longitude
                            lat_lons.append([lat, lon])
                            
                            popup_html = f"""
                            <div style="font-family: monospace; font-size: 11px; color: #1f2937;">
                                <b>{site.business_name}</b><br/>
                                ROI Score: {site.roi_rating}/10.0<br/>
                                Min. Inv: ₹ {site.min_investment:,.0f}<br/>
                                Est. Profit: ₹ {site.expected_profit:,.0f} / mo<br/>
                                ROI: {site.roi_percentage}%
                            </div>
                            """
                            
                            folium.Marker(
                                location=[lat, lon],
                                popup=folium.Popup(popup_html, max_width=250),
                                tooltip=f"Proposed Node {idx+1}: {site.business_name}",
                                icon=folium.Icon(color="cadetblue" if idx==0 else "blue", icon="shopping-cart", prefix="fa")
                            ).add_to(m)
                        
                        # Add connecting polyline if multiple nodes exist
                        if len(lat_lons) > 1:
                            folium.PolyLine(lat_lons, color="#14b8a6", weight=2, dash_array="5, 5").add_to(m)
                            
                        # Render Folium widget
                        st_folium(m, width=700, height=500, returned_objects=[])
                    else:
                        st.warning("No sites were returned by the optimization engine.")
                        
                with col2:
                    st.markdown("#### 📊 Evolved Plan Summary")
                    st.metric("Total Capital Expenditure Needed", f"₹ {payload.total_estimated_capex:,.0f}")
                    
                    st.markdown(f"**Evolved Plan Title:** `{payload.portfolio_name}`")
                    st.success(f"**Verification Status:** {payload.allocation_status}")
                    
                    st.markdown("---")
                    st.markdown("#### 🔍 Recommended Site Alternatives")
                    
                    # Render detail panels for each location node
                    for idx, site in enumerate(sites):
                        with st.expander(f"⭐ Node 0{idx+1}: {site.business_name} (Score: {site.roi_rating}/10)"):
                            st.write(f"**Coordinates:** `{site.latitude}, {site.longitude}`")
                            st.write(f"**Min Investment:** ₹ {site.min_investment:,.0f}")
                            st.write(f"**Expected Profit:** ₹ {site.expected_profit:,.0f} / mo")
                            st.write(f"**Expected Return on Investment (ROI):** `{site.roi_percentage}%`")
                            st.info(f"**Demographic & Feasibility Analysis:**\n{site.rag_context_summary}")
                            
                # JSON block at footer
                st.markdown("---")
                st.markdown("#### 📜 Verifiable Schema Contract JSON Response Payload")
                st.json(payload.model_dump())
                
            except Exception as e:
                st.error(f"Execution failed: {str(e)}")