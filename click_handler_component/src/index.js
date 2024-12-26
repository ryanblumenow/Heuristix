import React from "react";
import ReactDOM from "react-dom";
import { withStreamlitConnection, StreamlitComponentBase } from "streamlit-component-lib";

class ClickHandlerComponent extends StreamlitComponentBase {
  sendToStreamlit = (page) => {
    // Check if the Streamlit connection exists
    if (!this.props.streamlit) {
      console.error("Streamlit connection is not available.");
      return;
    }

    console.log("Streamlit property:", this.props.streamlit);

    console.log("Sending to Streamlit:", page);
    this.props.streamlit.setComponentValue(page); // Send the clicked value to Streamlit
  };

  render() {
    return (
      <div style={{ display: "flex", justifyContent: "center", gap: "20px" }}>
        <div
          onClick={() => this.sendToStreamlit("Dwell")}
          style={{
            width: "100px",
            height: "200px",
            backgroundColor: "#67A",
            cursor: "pointer",
            textAlign: "center",
            color: "white",
            fontWeight: "bold",
            paddingTop: "10px",
          }}
        >
          Dwell
        </div>
        <div
          onClick={() => this.sendToStreamlit("Dwell 2")}
          style={{
            width: "100px",
            height: "300px",
            backgroundColor: "#A76",
            cursor: "pointer",
            textAlign: "center",
            color: "white",
            fontWeight: "bold",
            paddingTop: "10px",
          }}
        >
          Dwell 2
        </div>
        <div
          onClick={() => this.sendToStreamlit("Artificial Intelligence")}
          style={{
            width: "100px",
            height: "400px",
            backgroundColor: "#34A853",
            cursor: "pointer",
            textAlign: "center",
            color: "white",
            fontWeight: "bold",
            paddingTop: "10px",
          }}
        >
          AI
        </div>
      </div>
    );
  }
}

// Wrap the component with Streamlit connection
const WrappedComponent = withStreamlitConnection(ClickHandlerComponent);

// Render the component into the root element
ReactDOM.render(<WrappedComponent />, document.getElementById("root"));
