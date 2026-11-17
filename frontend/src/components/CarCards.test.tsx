import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import CarCards from "./CarCards";

describe("CarCards", () => {
  it("renders nothing without cars", () => {
    const { container } = render(<CarCards cars={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders a card per car with price, body, rating, NCAP and a safe external link", () => {
    render(<CarCards cars={[
      { id: "a", name: "Hyundai Creta", url: "https://www.carwale.com/hyundai-cars/creta/", image: "https://img/c.png",
        price: "₹11.00 L – ₹20.00 L", body: "SUV", rating: 4.456, ncap: 5, status: "launched" },
      { id: "b", name: "Tata Sierra", url: "https://www.carwale.com/tata-cars/sierra/", image: null, price: "₹15.00 L",
        status: "upcoming" },
    ]} />);
    const [creta, sierra] = screen.getAllByRole("link");
    expect(creta).toHaveAttribute("href", "https://www.carwale.com/hyundai-cars/creta/");
    expect(creta).toHaveAttribute("target", "_blank");
    expect(creta).toHaveAttribute("rel", "noreferrer");
    expect(screen.getByAltText("Hyundai Creta")).toHaveAttribute("src", "https://img/c.png");
    expect(screen.getByText("₹11.00 L – ₹20.00 L")).toBeInTheDocument();
    expect(screen.getByText("5★")).toBeInTheDocument();
    expect(screen.getByText("· 4.5/5")).toBeInTheDocument();
    expect(sierra).toHaveTextContent("🚗");
    expect(sierra).toHaveTextContent("upcoming");
    expect(creta).not.toHaveTextContent("launched");
  });
});
