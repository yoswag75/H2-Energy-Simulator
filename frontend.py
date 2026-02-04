import streamlit as st # type: ignore
import numpy as np
import plotly.graph_objects as go # type: ignore
import time
import json

# Import Windows-compatible VQE backend
try:
    from backend import ImprovedVQE, EnergyLandscapeMapper, MultiMoleculeComparator
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    st.error("⚠️ Backend not found!")
# Configure page
st.set_page_config(
    page_title="Quantum Molecule Explorer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(120deg, #00d4ff, #9d4edd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(120deg, #00d4ff, #9d4edd);
        color: white;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">⚛️ Quantum Molecule Explorer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">VQE Molecular Energy Simulator</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    molecule = st.selectbox(
        "Select Molecule",
        ["H₂ (Hydrogen)"],
        help="Choose molecular system"
    )
    
    # Parse molecule
    molecule_map = {
        "H₂ (Hydrogen)": ("H2", 0.5, 2.0, 0.735)
    }
    
    mol_str, min_dist, max_dist, default_dist = molecule_map[molecule]
    
    distance = st.slider(
        "Bond Distance (Å)", 
        min_dist, max_dist, default_dist, 0.01,
        help="Adjust molecular geometry"
    )
    
    st.divider()
    
    with st.expander("🔬 Advanced Settings"):
        ansatz_choice = st.selectbox(
            "Ansatz Type",
            ["Efficient (Best)", "Hardware", "Full Entanglement"],
            help="Circuit architecture - Efficient usually works best"
        )
        
        ansatz_map = {
            "Efficient (Best)": "efficient",
            "Hardware": "hardware",
            "Full Entanglement": "full"
        }
        ansatz_type = ansatz_map[ansatz_choice]
        
        optimizer = st.selectbox(
            "Optimizer",
            ["COBYLA (Recommended)", "SLSQP", "Powell", "L-BFGS-B"],
            help="Classical optimization algorithm"
        )
        optimizer = optimizer.split(" ")[0]  # Extract name
        
        max_iter = st.slider(
            "Max Iterations", 
            50, 400, 150, 10,
            help="More iterations = better accuracy. 150-200 is usually good."
        )
    
    st.divider()
    
    run_button = st.button("🚀 Run VQE Simulation", type="primary")
    
    st.divider()
    
    with st.expander(" Features"):
        st.write("""
        **Features:**
        1. Adaptive Ansatz Selection
        2. Quantum Error Mitigation  
        3. Energy Landscape Mapping
        4. Multi-Molecule Comparison
        5. Circuit Compression
        6. Uses pre-computed Hamiltonians.
        """)

# Main content
tab1, tab2, tab3 = st.tabs(["📊 Simulation", "🗺️ Landscape", "⚖️ Compare"])

with tab1:
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if run_button and BACKEND_AVAILABLE:
            # Progress tracking
            progress_bar = st.progress(0)
            status = st.empty()
            
            status.info(f"🔄 Initializing VQE for {mol_str}...")
            time.sleep(0.5)
            progress_bar.progress(20)
            
            # Run VQE
            try:
                vqe = ImprovedVQE(mol_str, distance)
                
                status.info("🔄 Running quantum optimization...")
                progress_bar.progress(40)
                
                result = vqe.run_vqe(
                    optimizer=optimizer,
                    max_iter=max_iter,
                    ansatz_type=ansatz_type
                )
                
                progress_bar.progress(100)
                status.success("✅ Simulation complete!")
                time.sleep(0.5)
                
                progress_bar.empty()
                status.empty()
                
                # Results
                st.success("🎉 VQE Simulation Completed!")
                
                # Metrics
                col_m1, col_m2 = st.columns(2)
                
                with col_m1:
                    st.metric("Final Energy", f"{result['energy']:.6f} Ha")
                    st.metric("Nuclear Repulsion Energy", f"{result['nuclear_repulsion']:.6f} Ha")
                with col_m2:
                    st.metric("Iterations", result['num_iterations'])
                # Check chemical accuracy
                    targets = {"H2": -1.137, "LiH": -7.882, "H2O": -75.98}
                    target_energy = targets.get(mol_str, 0)
                    error = abs(result['energy'] - target_energy)
                    error_mha = error * 1000  # Convert to milliHartree
                    if error_mha < 1.6:
                        st.metric("Accuracy", "✅ Chemical", delta=f"{error_mha:.3f} mHa")
                    elif error_mha < 10:
                        st.metric("Accuracy", "⚠️ Near", delta=f"{error_mha:.3f} mHa")
                    else:
                        st.metric("Accuracy", "❌ Far", delta=f"{error_mha:.3f} mHa")
                
                st.divider()
                
                # Energy convergence plot
                st.subheader("📈 Energy Convergence")
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(result['history']))),
                    y=result['history'],
                    mode='lines+markers',
                    name='VQE Energy',
                    line=dict(color='#00d4ff', width=3),
                    marker=dict(size=6, color='#9d4edd'),
                    fill='tozeroy',
                    fillcolor='rgba(0, 212, 255, 0.1)'
                ))
                
                # Ground state line
                target_energy = targets.get(mol_str, result['energy'])
                fig.add_hline(
                    y=target_energy,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Target Ground State"
                )
                
                fig.update_layout(
                    title="VQE Energy Optimization",
                    xaxis_title="Iteration",
                    yaxis_title="Energy (Hartree)",
                    template="plotly_white",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Insights
                st.subheader("🔍 Analysis")
                
                col_i1, col_i2 = st.columns(2)
                
                with col_i1:
                    convergence_rate = abs((result['history'][-1] - result['history'][0]) / result['history'][0])
                    st.info(f"""
                    **Final Energy**: {result['energy']:.6f} Ha
                    
                    **Target Energy**: {target_energy:.6f} Ha
                    
                    **Error**: {error_mha:.3f} mHa
                    
                    **Convergence**: {(1-convergence_rate)*100:.1f}%
                    
                    **Optimizer**: {optimizer}
                    """)
                
                with col_i2:
                    if error_mha < 1.6:
                        st.success("""
                        ✅ **Chemical Accuracy Achieved!**
                        
                        Your simulation is accurate enough for 
                        real quantum chemistry applications!
                        
                        Error < 1.6 mHa ✓
                        """)
                    elif error_mha < 10:
                        st.warning(f"""
                        ⚠️ **Close to Chemical Accuracy**
                        
                        Error: {error_mha:.3f} mHa
                        Target: < 1.6 mHa
                        
                        **Try:**
                        - Increase iterations to {max_iter * 2}
                        - Switch to SLSQP optimizer
                        - Use different initial point
                        """)
                    else:
                        st.error(f"""
                        ❌ **Low Accuracy**
                        
                        Error: {error_mha:.1f} mHa (Target: <1.6)
                        
                        **Solutions:**
                        - Set max_iter to 200+
                        - Try SLSQP optimizer
                        - Check bond distance value
                        - Run multiple times
                        """)
                
                # Download button
                st.download_button(
                    label="💾 Download Results (JSON)",
                    data=json.dumps(result, indent=2),
                    file_name=f"vqe_{mol_str}_{distance:.3f}A.json",
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Try reducing max_iter or choosing a different molecule.")

with tab2:
    st.header("🗺️ Energy Landscape Mapping")
    st.write("Explore how molecular energy changes with bond distance")
    
    col_l1, col_l2 = st.columns([1, 3])
    
    with col_l1:
        num_points = st.slider("Number of Points", 10, 50, 30, help="More points = more detailed but slower")
        
        if st.button("🔍 Map Landscape"):
            with st.spinner(f"Mapping {mol_str} energy landscape..."):
                try:
                    mapper = EnergyLandscapeMapper(mol_str)
                    
                    landscape_data = mapper.map_energy_surface(
                        distance_range=(0.5,1.3),
                        num_points=num_points
                    )
                    
                    # Store in session state
                    st.session_state['landscape'] = landscape_data
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col_l2:
        if 'landscape' in st.session_state:
            data = st.session_state['landscape']
            
            # Clean the data - ensure proper types
            distances = [float(d) for d in data['distances']]
            energies = [float(e) for e in data['energies']]
            
            # Create plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=distances,
                y=energies,
                mode='lines+markers',
                name='Energy Surface',
                line=dict(color='#00d4ff', width=4),
                marker=dict(
                    size=10,
                    color=energies,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Energy (Ha)")
                )
            ))
            
            # Mark equilibrium
            eq_dist = float(data['equilibrium'])
            eq_energy = float(data['ground_state'])
            
            fig.add_trace(go.Scatter(
                x=[eq_dist],
                y=[eq_energy],
                mode='markers',
                name='Equilibrium',
                marker=dict(size=20, color='red', symbol='star')
            ))
            
            fig.update_layout(
                title=f"{mol_str} Potential Energy Surface",
                xaxis_title="Bond Distance (Å)",
                yaxis_title="Energy (Hartree)",
                template="plotly_white",
                height=500,
                xaxis=dict(
                    range=[min(distances) - 0.1, max(distances) + 0.1]  # Proper range
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Info
            st.success(f"""
            🎯 **Equilibrium Configuration**
            
            - Distance: {eq_dist:.3f} Å
            - Ground State: {eq_energy:.6f} Ha
            - Points Calculated: {len(distances)}
            """)

with tab3:
    st.header("⚖️ Multi-Molecule Comparison")
    
    st.write("Compare different molecular configurations with H₂ @ 0.735 Å")
    
    # Configuration selection
    configs = []
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        if st.checkbox("H₂ @ 0.735 Å", value=True, disabled= True):
            configs.append(("H2", 0.735))
        if st.checkbox("H₂ @ 1.0 Å"):
            configs.append(("H2", 1.0))
    
    with col_c2:
        if st.checkbox("LiH @ 1.596 Å", value=True):
            configs.append(("LiH", 1.596))
        if st.checkbox("LiH @ 2.0 Å"):
            configs.append(("LiH", 2.0))
    
    if st.button("📊 Run Comparison") and len(configs) > 1:
        with st.spinner("Running comparison..."):
            try:
                comparator = MultiMoleculeComparator()
                results = comparator.compare_molecules(configs)
                
                # Create comparison chart
                molecules = [f"{r['molecule']}@{r['distance']:.2f}Å" for r in results.values()]
                energies = [r['energy'] for r in results.values()]
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=molecules,
                    y=energies,
                    marker=dict(
                        color=energies,
                        colorscale='Viridis',
                        showscale=True
                    ),
                    text=[f"{e:.4f}" for e in energies],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title="Molecular Ground State Energy Comparison",
                    xaxis_title="Configuration",
                    yaxis_title="Energy (Hartree)",
                    template="plotly_white",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Table
                st.subheader("📋 Detailed Results")
                import pandas as pd
                df = pd.DataFrame(results.values())
                st.dataframe(df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")


# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>⚛️ Quantum Molecule Explorer</p>
    <p style='font-size: 0.9rem;'>Built with ❤️ by YoSwag75</p>
</div>
""", unsafe_allow_html=True)